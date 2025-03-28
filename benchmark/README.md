<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Understanding FLOPs vs. MACs in Model Benchmarking

When evaluating the computational cost of deep learning models, two common metrics often arise: FLOPs and MACs. Understanding the distinction between them is crucial for accurate model [benchmarking](https://docs.ultralytics.com/modes/benchmark/) and comparison.

## 📊 Defining FLOPs and MACs

-   **FLOPs (Floating Point Operations):** This term refers to the total number of basic arithmetic operations (like addition, subtraction, multiplication, division) involving floating-point numbers that a model performs during inference. You can learn more about the technical definition on [Wikipedia](https://en.wikipedia.org/wiki/FLOPS).
-   **MACs (Multiply-Accumulate Operations):** A MAC operation combines a multiplication and an addition into a single step: `a <- a + (b * c)`. This is a very common operation in neural networks, particularly in [convolutional layers](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) and fully connected layers. Read more about the [Multiply–accumulate operation](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation).

Since one MAC involves both a multiplication and an addition, it's often considered equivalent to **two** FLOPs. This is why you might see FLOP counts that are roughly double the MAC count for the same model.

## 🤔 The Complexity of Counting Operations

Accurately counting every single operation in a modern neural network is complex. Consider a simple matrix multiplication: conceptually, it involves a series of multiplications and additions. However, how these operations are executed can vary significantly depending on:

-   **Implementation:** Different code implementations might structure the calculations differently.
-   **Hardware:** Specialized hardware like [GPUs](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) and TPUs use highly parallelized execution units ([CUDA cores](https://developer.nvidia.com/cuda-toolkit), Tensor Cores) that perform many operations simultaneously. See how [parallel computing](https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial) impacts performance.
-   **Compiler Optimizations:** Compilers often fuse operations or rearrange calculations to optimize for speed or efficiency.

A naive count based on the high-level mathematical definition might not reflect the actual computational workload on the hardware. Measuring true performance often requires careful [profiling and benchmarking](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations).

## 💡 The `thop` Approach: Focusing on Multiplications

When comparing different model architectures or seeking an **implementation-agnostic** measure of computational complexity, a consistent metric is needed. The `thop` library (and similar tools) often simplifies the calculation by primarily focusing on the number of **multiplications**.

Why multiplications?
-   They are typically the most computationally intensive part of operations like convolutions and matrix multiplies.
-   Their count is less sensitive to minor implementation details compared to counting every single addition, activation function, etc.

Therefore, in `thop`:
-   **MACs** generally represent the count of multiply-accumulate operations, often dominated by multiplications.
-   **FLOPs** are often *approximated* by multiplying the MAC count by two, based on the common `1 MAC ≈ 2 FLOPs` heuristic.

This approach provides a standardized way to compare models based on their core computational demands, aiding in [model optimization](https://www.ultralytics.com/glossary/model-optimization) efforts. For detailed performance analysis on specific hardware, direct [benchmarking](https://docs.ultralytics.com/modes/benchmark/) is still recommended. Explore resources like [understanding GPU performance](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) for deeper insights.

## 🤝 Contributing

Contributions to improve documentation or the library itself are welcome! Please see the main Ultralytics repository guidelines for contributing: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). We appreciate your help in making our tools and resources better for the community. Check out our [contribution guide](https://docs.ultralytics.com/help/contributing/) for more details.
