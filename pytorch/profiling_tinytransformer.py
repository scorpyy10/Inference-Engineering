
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as profiler
import math

# ============================================================
# 1. CONFIGURATION
# ============================================================

torch.manual_seed(42)

# Use one CPU thread for controlled comparisons
torch.set_num_threads(1)

print("PyTorch version:", torch.__version__)
print("Device: CPU")


# ============================================================
# 2. TINY TRANSFORMER BLOCK
# ============================================================

class TinyTransformerBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        # 1. Self Attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])

        weights = F.softmax(scores, dim=-1)

        attention = weights @ v

        # 2. Residual Connection + LayerNorm
        x = self.norm1(x + attention)

        # 3. Feed Forward Network
        ff_out = F.relu(self.ff1(x))
        ff_out = self.ff2(ff_out)

        # 4. Residual Connection + LayerNorm
        out = self.norm2(x + ff_out)

        return out, weights


# ============================================================
# 3. EXPERIMENT CONFIGURATIONS
# ============================================================

# Format: (Experiment, Sequence Length, d_model, d_ff)

experiments = [
    ("Current", 4, 4, 8),
    ("A", 32, 64, 128),
    ("B", 128, 128, 256),
    ("C", 512, 256, 512),
]

B = 1  # Batch size


# ============================================================
# 4. RUN EXPERIMENTS
# ============================================================

for name, S, D, FF in experiments:

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    # Generate random input
    # Shape: [Batch, Sequence Length, Embedding Dimension]

    x = torch.randn(B, S, D)

    # Initialize model
    model = TinyTransformerBlock(
        d_model=D,
        d_ff=FF
    )

    model.eval()

    # Count trainable parameters
    params = sum(
        p.numel() for p in model.parameters()
    )

    print("\n========== MODEL CONFIGURATION ==========")
    print("Input shape:", x.shape)
    print("Batch size:", B)
    print("Sequence length:", S)
    print("d_model:", D)
    print("d_ff:", FF)
    print(f"Trainable parameters: {params:,}")

    # --------------------------------------------------------
    # 5. WARMUP
    # --------------------------------------------------------

    with torch.inference_mode():

        for _ in range(20):
            output, attention_weights = model(x)

    # --------------------------------------------------------
    # 6. PROFILE MODEL
    # --------------------------------------------------------

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU
        ],
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:

        with torch.inference_mode():

            for _ in range(50):
                output, attention_weights = model(x)

    # --------------------------------------------------------
    # 7. TOP OPERATORS
    # --------------------------------------------------------

    print("\n========== TOP 10 OPERATORS ==========")

    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=10
        )
    )

    # --------------------------------------------------------
    # 8. OPERATOR BREAKDOWN BY INPUT SHAPE
    # --------------------------------------------------------

    print("\n========== OPERATORS BY INPUT SHAPE ==========")

    print(
        prof.key_averages(
            group_by_input_shape=True
        ).table(
            sort_by="self_cpu_time_total",
            row_limit=15
        )
    )

    # --------------------------------------------------------
    # 9. OUTPUT SHAPES
    # --------------------------------------------------------

    print("\n========== OUTPUT SHAPES ==========")

    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Attention weights:", attention_weights.shape)

    # --------------------------------------------------------
    # 10. PROFILING SUMMARY
    # --------------------------------------------------------

    total_self_cpu_time = sum(
        event.self_cpu_time_total
        for event in prof.key_averages()
    )

    avg_self_cpu_time = total_self_cpu_time / 50

    print("\n========== PROFILING SUMMARY ==========")

    print(
        f"Total self CPU time: "
        f"{total_self_cpu_time / 1000:.3f} ms"
    )

    print(
        f"Average operator self CPU time per pass: "
        f"{avg_self_cpu_time:.2f} us"
    )

    # --------------------------------------------------------
    # 11. CLEANUP
    # --------------------------------------------------------

    del model
    del x
    del output
    del attention_weights

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 70)
