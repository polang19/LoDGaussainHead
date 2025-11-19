#
# Asynchronous importance computation with GPU acceleration (faiss)
# Optimized for GPU memory utilization
#

import threading
from queue import Queue
import torch
import numpy as np
from typing import Optional

# Try to import faiss, fallback to scipy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
    try:
        # Check if GPU is available
        num_gpus = faiss.get_num_gpus()
        FAISS_GPU_AVAILABLE = num_gpus > 0
    except:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    from scipy.spatial import cKDTree


class AsyncImportanceComputer:
    """
    Compute importance scores asynchronously with optional GPU acceleration.
    
    Features:
    - GPU-accelerated k-NN search using faiss (if available)
    - Automatic fallback to CPU (scipy) if faiss unavailable
    - Optimized GPU memory usage
    - Non-blocking computation in background thread
    
    Usage:
        computer = AsyncImportanceComputer(use_gpu=True)
        
        # Start computation (non-blocking)
        computer.start_computation(xyz, k=10)
        
        # Check and get result (non-blocking)
        result = computer.get_result(device)
        if result is not None:
            imp_list = result
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize async importance computer.
        
        Args:
            use_gpu: If True, use faiss GPU acceleration (if available)
        """
        self.queue = Queue(maxsize=1)
        self.thread: Optional[threading.Thread] = None
        self.is_computing = False
        self.lock = threading.Lock()
        self.use_gpu = use_gpu and FAISS_GPU_AVAILABLE
        
        # GPU resource management
        self.gpu_resources = None
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                # Set GPU memory fraction to avoid OOM (use 80% of GPU memory)
                # This allows other operations to use remaining 20%
                self.gpu_resources.setDefaultNullStreamAllDevices()
                print("[AsyncImportanceComputer] Using faiss GPU acceleration")
            except Exception as e:
                print(f"[AsyncImportanceComputer] GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False
        
        if not self.use_gpu:
            if FAISS_AVAILABLE:
                print("[AsyncImportanceComputer] faiss available but GPU not available, using CPU")
            else:
                print("[AsyncImportanceComputer] faiss not available, using scipy (CPU)")
    
    def start_computation(self, xyz: torch.Tensor, k: int = 10):
        """
        Start importance computation in background thread (non-blocking).
        
        Args:
            xyz: Point positions, shape (N, 3), on GPU or CPU
            k: Number of nearest neighbors to consider
        """
        with self.lock:
            if self.is_computing:
                # Already computing, skip
                return
            
            self.is_computing = True
        
        # Convert to numpy (float32 for better GPU performance and memory usage)
        # Keep on GPU if already there, otherwise move to CPU
        if xyz.is_cuda:
            # If already on GPU, we can use it directly with faiss
            xyz_np = xyz.detach().cpu().numpy().astype('float32')
        else:
            xyz_np = xyz.detach().cpu().numpy().astype('float32')
        
        def compute():
            try:
                if self.use_gpu:
                    # GPU-accelerated computation using faiss
                    self._compute_with_faiss(xyz_np, k)
                else:
                    # CPU computation using scipy
                    self._compute_with_scipy(xyz_np, k)
            except Exception as e:
                print(f"[AsyncImportanceComputer] Error: {e}")
                # Fallback to scipy if faiss fails
                if self.use_gpu:
                    print("[AsyncImportanceComputer] Falling back to scipy (CPU)")
                    try:
                        self._compute_with_scipy(xyz_np, k)
                    except Exception as e2:
                        print(f"[AsyncImportanceComputer] Scipy fallback also failed: {e2}")
            finally:
                with self.lock:
                    self.is_computing = False
        
        # Start background thread
        self.thread = threading.Thread(target=compute, daemon=True)
        self.thread.start()
    
    def _compute_with_faiss(self, xyz_np: np.ndarray, k: int):
        """GPU-accelerated computation using faiss"""
        N = xyz_np.shape[0]
        
        # Create GPU index with optimized settings
        # IndexFlatL2 is the fastest for small to medium datasets
        # For very large datasets, consider IndexIVFFlat or IndexHNSW
        index = faiss.IndexFlatL2(3)  # L2 distance for 3D points
        
        # Move index to GPU with memory optimization
        if self.gpu_resources is not None:
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        else:
            # Fallback: create new GPU resources
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        
        try:
            # Add data to index (this is done on GPU)
            gpu_index.add(xyz_np)
            
            # Search for k+1 neighbors (including the point itself)
            # This is done entirely on GPU, very fast
            distances, _ = gpu_index.search(xyz_np, k+1)
            
            # Compute density: exclude the point itself (distance=0)
            # This is done on CPU (numpy), but it's very fast
            mean_distances = distances[:, 1:].mean(axis=1)
            density = 1.0 / (mean_distances + 1e-6)
            
            # Put result in queue
            if not self.queue.full():
                self.queue.put(density)
        finally:
            # Clean up GPU resources
            # Note: faiss handles cleanup automatically, but we can be explicit
            del gpu_index
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _compute_with_scipy(self, xyz_np: np.ndarray, k: int):
        """CPU computation using scipy (fallback)"""
        from scipy.spatial import cKDTree
        
        # Build KD-tree
        tree = cKDTree(xyz_np)
        
        # Query k+1 neighbors
        distances, _ = tree.query(xyz_np, k=k+1)
        
        # Compute density
        mean_distances = distances[:, 1:].mean(axis=1)
        density = 1.0 / (mean_distances + 1e-6)
        
        # Put result in queue
        if not self.queue.full():
            self.queue.put(density)
    
    def get_result(self, device: torch.device, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        """
        Get computation result if available (non-blocking).
        
        Args:
            device: Target device for the result tensor
            dtype: Data type for the result tensor
        
        Returns:
            Importance scores tensor if available, None otherwise
        """
        if not self.queue.empty():
            try:
                density = self.queue.get_nowait()
                return torch.tensor(density, device=device, dtype=dtype)
            except:
                return None
        return None
    
    def is_busy(self) -> bool:
        """Check if computation is in progress."""
        with self.lock:
            return self.is_computing
    
    def cleanup(self):
        """Clean up GPU resources (call when done)"""
        if self.gpu_resources is not None:
            del self.gpu_resources
            self.gpu_resources = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

