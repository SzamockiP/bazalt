#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// What bazalt reads out of a compiled shader, and nothing more.
//
// This exists to answer ONE question the automatic barrier tracker could not
// answer before 0.19: does this shader WRITE the resource at (set, binding)?
// Without it, a storage buffer bound to a graphics pipeline was assumed read-only
// and a storage image bound to one was not tracked at all — so a fragment
// `imageStore` needed a hand-written cmd.barrier() or it was a validation error.
// It also answers two smaller questions that were accepted ceilings: does the
// shader print, and did an HLSL entry point resolve to an empty function.
//
// Deliberately NOT a general reflection library. It does not report the bindings a
// shader declares, because bazalt takes those from the pipeline builder and a
// second source would disagree with the first at hot-reload time (see DESIGN.md).
// It knows nothing about ShaderStage either: it reports raw SPIR-V execution
// models and ShaderCompiler maps them, so the switches that must break when a
// stage is added all stay in one file.
//
// ── The invariant, and the reason to trust the result ──────────────────────────
//
// READS ARE ALWAYS ASSUMED. WRITES ARE ONLY EVER NARROWED BY A POSITIVE PROOF OF
// ABSENCE.
//
// Every uncertainty — a descriptor handed to a function, a pointer phi, a decoration
// group, a malformed word, or SPIR-V bazalt did not compile — sets
// `writes_unknown`, and writes() then answers true for everything. An uncertain
// module therefore behaves exactly as bazalt did before this file existed:
// conservative. The only way to LOSE correctness is a parser bug that positively
// claims "no write" where there is one, which is the narrow class the tables below
// are built to keep small.
struct ShaderReflection
{
    // Sorted and unique. Empty means "provably writes no descriptor", which is
    // what lets the tracker drop a barrier.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> written_bindings;

    // Raw SPIR-V ExecutionModel values (Vertex=0, TessellationControl=1,
    // TessellationEvaluation=2, Geometry=3, Fragment=4, GLCompute=5). A module may
    // declare several: multi-entry-point SPIR-V is legal.
    std::vector<std::uint32_t> execution_models;

    // The module imports NonSemantic.DebugPrintf, so it contains debugPrintfEXT
    // calls. Must be read off UNOPTIMIZED words: a print has no observable result,
    // so the optimizer is entitled to delete it.
    bool prints = false;

    // The requested entry point exists but its body does nothing and it declares
    // no interface. glslang synthesizes an entry point under a name that matches no
    // function, so an HLSL typo compiles to a shader that draws nothing.
    bool empty_entry_point = false;

    // Set by anything this parser cannot follow. See the invariant above.
    bool writes_unknown = false;

    bool writes(std::uint32_t set, std::uint32_t binding) const
    {
        if (writes_unknown)
        {
            return true;
        }
        return std::ranges::binary_search(written_bindings, std::pair{set, binding});
    }

    bool declares_execution_model(std::uint32_t model) const
    {
        return std::ranges::find(execution_models, model) != execution_models.end();
    }
};

namespace spirv_detail
{
    // Only the opcodes this parser acts on. Named rather than inlined so the tables
    // below read as data.
    enum Op : std::uint32_t
    {
        kExtInstImport = 11,
        kEntryPoint = 15,
        kDecorate = 71,
        kDecorationGroup = 73,
        kGroupDecorate = 74,
        kGroupMemberDecorate = 75,
        kFunction = 54,
        kFunctionEnd = 56,
        kFunctionCall = 57,
        kPhi = 245,
        kLabel = 248,
        kReturn = 253,
    };

    // Decoration values.
    constexpr std::uint32_t kDecorationBinding = 33;
    constexpr std::uint32_t kDecorationDescriptorSet = 34;

    // Instructions that produce a new id standing for the same underlying object.
    // Every one of them has the same operand layout — result type at +1, result id at
    // +2, the thing it aliases at +3 — which is what makes the walk cheap.
    //
    // OpImageTexelPointer is in here and it is load-bearing: it is how
    // imageAtomicAdd() reaches a storage image, so a table without it reports an
    // atomically-incremented image as read-only.
    constexpr std::uint32_t kAliasOps[] = {
        60,  // OpImageTexelPointer
        61,  // OpLoad
        65,  // OpAccessChain
        66,  // OpInBoundsAccessChain
        67,  // OpPtrAccessChain
        70,  // OpInBoundsPtrAccessChain
        83,  // OpCopyObject
        124, // OpBitcast
        400, // OpCopyLogical
    };

    // Instructions that write through a pointer given as their FIRST operand.
    constexpr std::uint32_t kWriteOpsPointerAt1[] = {
        62,  // OpStore
        63,  // OpCopyMemory
        64,  // OpCopyMemorySized
        99,  // OpImageWrite
        228, // OpAtomicStore
        319, // OpAtomicFlagClear
    };

    // Instructions that write through a pointer given as their THIRD operand, because
    // they return a value and so carry a result type and result id first.
    //
    // The atomics are the reason this list exists. A GPU counter incremented with
    // atomicAdd() and never stored to is the ordinary case, and a scan that looked only
    // for OpStore would report its buffer as read-only — silently reinstating the exact
    // debt this file pays off.
    constexpr std::uint32_t kWriteOpsPointerAt3[] = {
        229,  // OpAtomicExchange
        230,  // OpAtomicCompareExchange
        231,  // OpAtomicCompareExchangeWeak
        232,  // OpAtomicIIncrement
        233,  // OpAtomicIDecrement
        234,  // OpAtomicIAdd
        235,  // OpAtomicISub
        236,  // OpAtomicSMin
        237,  // OpAtomicUMin
        238,  // OpAtomicSMax
        239,  // OpAtomicUMax
        240,  // OpAtomicAnd
        241,  // OpAtomicOr
        242,  // OpAtomicXor
        318,  // OpAtomicFlagTestAndSet
        5614, // OpAtomicFMinEXT
        5615, // OpAtomicFMaxEXT
        6035, // OpAtomicFAddEXT
    };
    // OpAtomicLoad (227) is deliberately absent from both: it is the one atomic that
    // only reads, and a buffer touched by nothing else is genuinely read-only.

    // There is deliberately NO capability allowlist here.
    //
    // The first version had one, to catch a module whose instruction set this parser
    // has not been taught — cooperative matrix, ray tracing, PhysicalStorageBuffer
    // addressing all write memory through opcodes the tables above do not list. It was
    // the wrong mechanism: the capability numbers are a long enum, an allowlist written
    // from memory is wrong in both directions, and being wrong the safe way (flagging a
    // capability that is fine) silently turns the whole feature off while looking like
    // it works.
    //
    // The real question is not "which capabilities" but "did bazalt compile this?".
    // The write set for GLSL and HLSL going through shaderc for a Vulkan 1.2/1.3 target
    // is exactly what the two tables above cover. Foreign SPIR-V — a .spv file, or
    // compile_shader(source=bytes) — is the only way an unlisted write opcode arrives,
    // and provenance is something the CALLER knows for certain and this parser cannot
    // see at all. So ShaderCompiler sets writes_unknown on those two paths, and this
    // file stays about parsing.

    inline bool contains(std::span<const std::uint32_t> table, std::uint32_t value)
    {
        return std::ranges::find(table, value) != table.end();
    }
} // namespace spirv_detail

// Walk a SPIR-V module once and report the facts above.
//
// `entry_point` selects which function the empty-body check looks at; empty means
// the language default and turns that check off, because a GLSL main() that
// deliberately does nothing is legitimate.
//
// ── Why one forward pass is enough ────────────────────────────────────────────
//
// SPIR-V's own ordering rules do the work that would otherwise need a second pass:
//
//  * Decorations live in logical layout section 7 and every function definition in
//    section 11, so the id -> (set, binding) map is COMPLETE before the first
//    OpStore is ever seen.
//  * Inside a function, SSA dominance plus the rule that a block precedes every
//    block it dominates mean an OpAccessChain always appears earlier in the word
//    stream than the OpStore consuming it. So resolving a pointer to its root
//    variable is one map lookup, never a chase and never a fixpoint.
//  * OpEntryPoint (section 4) precedes OpFunction (section 11), so the
//    name -> function id -> body order also works in one pass.
//
// The cases where that is NOT enough are enumerated in the body, and each one sets
// writes_unknown rather than guessing.
inline ShaderReflection reflect_spirv(std::span<const std::uint32_t> words, std::string_view entry_point = {})
{
    using namespace spirv_detail;

    ShaderReflection result;

    // A SPIR-V module starts with a 5-word header. Anything shorter is not one.
    constexpr std::size_t kHeaderWords = 5;
    if (words.size() < kHeaderWords)
    {
        result.writes_unknown = true;
        return result;
    }

    // id -> (set, binding), for ids carrying BOTH decorations. That pair is what
    // makes a variable a descriptor, so no type walk and no OpVariable handling is
    // needed at all.
    std::unordered_map<std::uint32_t, std::uint32_t> descriptor_set_of;
    std::unordered_map<std::uint32_t, std::uint32_t> binding_of;
    // id -> the id it ultimately stands for.
    std::unordered_map<std::uint32_t, std::uint32_t> root_of;
    std::unordered_set<std::uint32_t> written_roots;

    const auto resolve = [&](std::uint32_t id)
    {
        auto it = root_of.find(id);
        return it == root_of.end() ? id : it->second;
    };
    const auto mark_written = [&](std::uint32_t pointer) { written_roots.insert(resolve(pointer)); };

    // Which function id the requested entry point names, and whether we are inside
    // it. Zero means "not found yet".
    std::uint32_t entry_function = 0;
    bool entry_declared_interface = false;
    bool entry_body_is_trivial = true;
    bool inside_entry_body = false;

    std::size_t i = kHeaderWords;
    while (i < words.size())
    {
        const std::uint32_t opcode = words[i] & 0xFFFFu;
        const std::uint32_t count = words[i] >> 16;

        // A zero word count would loop forever, and an instruction running past
        // the end is a truncated module. Neither is something to guess through.
        if (count == 0 || i + count > words.size())
        {
            result.writes_unknown = true;
            break;
        }

        const auto operand = [&](std::size_t n) -> std::uint32_t { return (n < count) ? words[i + n] : 0u; };

        switch (opcode)
        {
            case kExtInstImport:
                if (count > 2)
                {
                    // The name is a packed, NUL-terminated literal string starting
                    // at word 2.
                    const char* name = reinterpret_cast<const char*>(&words[i + 2]);
                    const std::size_t bytes = (count - 2) * sizeof(std::uint32_t);
                    if (std::string_view(name, bytes).starts_with("NonSemantic.DebugPrintf"))
                    {
                        result.prints = true;
                    }
                }
                break;

            case kEntryPoint:
            {
                result.execution_models.push_back(operand(1));
                if (!entry_point.empty() && count > 3)
                {
                    const char* name = reinterpret_cast<const char*>(&words[i + 3]);
                    const std::size_t bytes = (count - 3) * sizeof(std::uint32_t);
                    const std::string_view packed(name, bytes);
                    const std::size_t nul = packed.find('\0');
                    const std::string_view declared = packed.substr(0, nul == std::string_view::npos ? bytes : nul);
                    if (declared == entry_point)
                    {
                        entry_function = operand(2);
                        // Every id after the packed name is an interface variable.
                        // A shader with no inputs and no outputs at all is the
                        // shape glslang synthesizes for a name matching nothing.
                        const std::size_t name_words =
                            (nul == std::string_view::npos) ? (count - 3) : (nul / sizeof(std::uint32_t) + 1);
                        entry_declared_interface = (3 + name_words) < count;
                    }
                }
                break;
            }

            case kDecorate:
                if (count >= 4)
                {
                    if (operand(2) == kDecorationDescriptorSet)
                    {
                        descriptor_set_of[operand(1)] = operand(3);
                    }
                    else if (operand(2) == kDecorationBinding)
                    {
                        binding_of[operand(1)] = operand(3);
                    }
                }
                break;

            // Decoration groups would make the two maps above incomplete. glslang
            // never emits them, so treating them as unknown costs nothing real.
            case kDecorationGroup:
            case kGroupDecorate:
            case kGroupMemberDecorate:
                result.writes_unknown = true;
                break;

            case kFunction:
                inside_entry_body = (entry_function != 0 && operand(2) == entry_function);
                break;

            case kFunctionEnd:
                inside_entry_body = false;
                break;

            case kFunctionCall:
                // At -O0 nothing is inlined, and the callee may be DEFINED LATER in
                // the word stream — so a one-pass walk cannot know what it does.
                // Any descriptor handed to a function is therefore assumed written.
                // Pessimism confined to descriptors actually passed as arguments.
                for (std::size_t n = 4; n < count; ++n)
                {
                    const std::uint32_t root = resolve(words[i + n]);
                    if (descriptor_set_of.contains(root) || binding_of.contains(root))
                    {
                        written_roots.insert(root);
                    }
                }
                break;

            case kPhi:
                // A phi can name a value defined in a later block across a loop
                // back edge, so the same reasoning applies. Pointer phis need the
                // VariablePointers capability and are rare.
                for (std::size_t n = 3; n < count; ++n)
                {
                    const std::uint32_t root = resolve(words[i + n]);
                    if (descriptor_set_of.contains(root) || binding_of.contains(root))
                    {
                        written_roots.insert(root);
                    }
                }
                break;

            default:
                if (contains(kAliasOps, opcode) && count >= 4)
                {
                    root_of[operand(2)] = resolve(operand(3));
                }
                else if (contains(kWriteOpsPointerAt1, opcode) && count >= 2)
                {
                    mark_written(operand(1));
                }
                else if (contains(kWriteOpsPointerAt3, opcode) && count >= 4)
                {
                    mark_written(operand(3));
                }
                break;
        }

        // Anything in the entry point's body beyond a label and a return means the
        // function does something. Checked after the switch so the instruction has
        // already been classified.
        if (inside_entry_body && opcode != kFunction && opcode != kLabel && opcode != kReturn)
        {
            entry_body_is_trivial = false;
        }

        i += count;
    }

    result.empty_entry_point = (entry_function != 0) && entry_body_is_trivial && !entry_declared_interface;

    // Turn written ids into (set, binding) pairs. An id decorated with only one of
    // the two is not a descriptor, so it is not reported.
    result.written_bindings.reserve(written_roots.size());
    for (std::uint32_t id : written_roots)
    {
        const auto set_it = descriptor_set_of.find(id);
        const auto binding_it = binding_of.find(id);
        if (set_it != descriptor_set_of.end() && binding_it != binding_of.end())
        {
            result.written_bindings.emplace_back(set_it->second, binding_it->second);
        }
    }
    std::ranges::sort(result.written_bindings);
    const auto last = std::ranges::unique(result.written_bindings);
    result.written_bindings.erase(last.begin(), last.end());

    return result;
}
