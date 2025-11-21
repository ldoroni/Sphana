using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;

namespace Sphana.Database.Infrastructure.Onnx;

/// <summary>
/// Base class for ONNX model inference with pooling and batching support
/// </summary>
public abstract class OnnxModelBase : IDisposable
{
    private readonly SemaphoreSlim _sessionPoolSemaphore;
    private readonly ConcurrentBag<InferenceSession> _sessionPool;
    private readonly Microsoft.ML.OnnxRuntime.SessionOptions _sessionOptions;
    protected readonly ILogger _logger;
    protected readonly string _modelPath;
    protected readonly int _maxPoolSize;
    protected readonly bool _useGpu;

    protected OnnxModelBase(
        string modelPath,
        bool useGpu,
        int gpuDeviceId,
        int maxPoolSize,
        ILogger logger)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        _useGpu = useGpu;
        _maxPoolSize = maxPoolSize;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _sessionOptions = CreateSessionOptions(useGpu, gpuDeviceId);
        _sessionPool = new ConcurrentBag<InferenceSession>();
        _sessionPoolSemaphore = new SemaphoreSlim(maxPoolSize, maxPoolSize);

        // Pre-warm the pool with one session
        try
        {
            var session = CreateSession();
            _sessionPool.Add(session);
            _logger.LogInformation("ONNX model loaded successfully: {ModelPath}, GPU: {UseGpu}", 
                modelPath, useGpu);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize ONNX model: {ModelPath}", modelPath);
            throw;
        }
    }

    private Microsoft.ML.OnnxRuntime.SessionOptions CreateSessionOptions(bool useGpu, int gpuDeviceId)
    {
        var options = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        if (useGpu)
        {
            try
            {
                // Try to use CUDA execution provider
                options.AppendExecutionProvider_CUDA(gpuDeviceId);
                _logger.LogInformation("CUDA execution provider enabled on device {DeviceId}", gpuDeviceId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, 
                    "Failed to enable CUDA execution provider, falling back to CPU. " +
                    "Ensure CUDA, cuDNN, and compatible ONNX Runtime GPU are installed.");
                // Will fall back to CPU automatically
            }
        }

        return options;
    }

    private InferenceSession CreateSession()
    {
        if (!File.Exists(_modelPath))
        {
            throw new FileNotFoundException($"ONNX model file not found: {_modelPath}");
        }

        return new InferenceSession(_modelPath, _sessionOptions);
    }

    protected async Task<InferenceSession> AcquireSessionAsync(CancellationToken cancellationToken = default)
    {
        await _sessionPoolSemaphore.WaitAsync(cancellationToken);

        if (_sessionPool.TryTake(out var session))
        {
            return session;
        }

        // Create new session if pool is not full
        return CreateSession();
    }

    protected void ReleaseSession(InferenceSession session)
    {
        _sessionPool.Add(session);
        _sessionPoolSemaphore.Release();
    }

    protected static Tensor<float> CreateTensor(float[][] data, int[] dimensions)
    {
        var tensor = new DenseTensor<float>(dimensions);
        var flatIndex = 0;
        for (int i = 0; i < data.Length; i++)
        {
            for (int j = 0; j < data[i].Length; j++)
            {
                tensor.SetValue(flatIndex++, data[i][j]);
            }
        }
        return tensor;
    }

    protected static Tensor<long> CreateTensor(long[][] data, int[] dimensions)
    {
        var tensor = new DenseTensor<long>(dimensions);
        var flatIndex = 0;
        for (int i = 0; i < data.Length; i++)
        {
            for (int j = 0; j < data[i].Length; j++)
            {
                tensor.SetValue(flatIndex++, data[i][j]);
            }
        }
        return tensor;
    }

    public virtual void Dispose()
    {
        while (_sessionPool.TryTake(out var session))
        {
            session?.Dispose();
        }

        _sessionOptions?.Dispose();
        _sessionPoolSemaphore?.Dispose();
    }
}

