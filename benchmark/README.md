<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Understanding FLOPs vs. MACs in Model Benchmarking

When evaluating the computational cost of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models like [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), two common metrics often arise: FLOPs and MACs. Understanding the distinction between them is crucial for accurate model [benchmarking](https://docs.ultralytics.com/modes/benchmark/) and comparison, helping you choose the right model for your hardware and latency requirements.

## 📊 Defining FLOPs and MACs

- **FLOPs (Floating Point Operations):** This term refers to the total number of basic arithmetic operations (like addition, subtraction, multiplication, division) involving floating-point numbers that a model performs during a single forward pass (inference). You can learn more about the technical definition from [Wikipedia's FLOPS page](https://en.wikipedia.org/wiki/FLOPS).
- **MACs (Multiply-Accumulate Operations):** A MAC operation combines a multiplication and an addition into a single step: `a <- a + (b * c)`. This is a fundamental and frequent operation within [neural networks](https://www.ultralytics.com/glossary/neural-network-nn), especially prevalent in [convolutional layers](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) and fully connected layers. Read more about the [Multiply–accumulate operation](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation).

Since one MAC involves both a multiplication and an addition, it's often considered roughly equivalent to **two** FLOPs. This heuristic is why you might observe FLOP counts that are approximately double the MAC count reported for the same model architecture.

## 🤔 The Complexity of Counting Operations

Accurately counting every single arithmetic operation in a modern neural network is surprisingly complex. Consider a simple matrix multiplication: conceptually, it involves a defined series of multiplications and additions. However, the actual execution can vary significantly depending on several factors:

- **Implementation Details:** Different software libraries or custom code might structure the calculations differently, potentially leading to variations in the exact operation count.
- **Hardware Acceleration:** Specialized hardware like [GPUs](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) and TPUs utilize highly parallel architectures ([CUDA cores](https://developer.nvidia.com/cuda-toolkit), Tensor Cores) that perform numerous operations simultaneously. The way operations map to these units affects the real computational time, which might not directly correlate with a simple FLOP count. See how [parallel computing](https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial) impacts performance.
- **Compiler Optimizations:** Modern compilers often apply sophisticated optimizations, such as fusing multiple operations (like convolution, bias addition, and activation) into a single kernel or rearranging calculations to enhance speed or memory efficiency.

Therefore, a naive operation count based solely on the high-level mathematical definition of the network layers might not accurately reflect the actual computational workload or the resulting [inference latency](https://www.ultralytics.com/glossary/inference-latency) on specific hardware. Measuring true performance often requires careful profiling using tools like the [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) or [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler) and direct [benchmarking](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations).

## 💡 The `thop` Approach: Focusing on Multiplications

When comparing different model architectures or seeking an **implementation-agnostic** measure of computational complexity, a consistent and standardized metric is essential. The `thop` library (and similar tools) often simplifies the calculation by primarily focusing on the number of **multiplications**, particularly within MAC operations.

Why focus on multiplications?

- They are typically the most computationally intensive part of the core operations in deep learning, such as convolutions and matrix multiplies.
- Their count tends to be less sensitive to minor implementation variations (e.g., choice of activation function, specific bias addition methods) compared to counting every single addition, activation function computation, or normalization step.

Therefore, in the context of `thop` and similar benchmarking tools:

- **MACs** generally represent the count of multiply-accumulate operations, with the count heavily influenced by the number of multiplications in layers like `Conv2d` and `Linear`.
- **FLOPs** are often _approximated_ by multiplying the calculated MAC count by two, adhering to the common `1 MAC ≈ 2 FLOPs` heuristic.

This approach provides a standardized and reproducible way to compare the theoretical computational demands of different models, aiding in architectural design choices and high-level efficiency analysis. However, for precise performance evaluation and optimization on specific target hardware, direct [benchmarking](https://docs.ultralytics.com/modes/benchmark/) remains the most reliable method. Explore resources like [understanding GPU performance for deep learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) for deeper insights into hardware-specific considerations.

## 🤝 Contributing

Contributions to improve this documentation or the underlying library are always welcome! If you have suggestions or find inaccuracies, please refer to the main Ultralytics repository guidelines for contributing at [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). We appreciate your help in making our tools and resources better for the entire computer vision community. Check out our [contribution guide](https://docs.ultralytics.com/help/contributing/) for more detailed information on how to get involved.
