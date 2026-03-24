import dpnp as dp
import dpctl
import numpy as np
import threading
import queue
import mytorch

class AsyncDataLoader:
    def __init__(
        self,
        dataset,
        batch_size=64,
        num_workers=4,
        prefetch=8,
        device="gpu"
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.queue = queue.Queue(prefetch)
        self.index_queue = queue.Queue()
        self.stop_signal = object()
        self.sycl_queue = dp.array([1], device=device).sycl_queue

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def _worker(self, index_queue):
        while True:
            batch_indices = index_queue.get()
            if batch_indices is None:
                break

            xs, ys = self.dataset[batch_indices]

            # Allocate device USM memory
            x_usm = dpctl.tensor.asarray(
                xs,
                sycl_queue=self.sycl_queue
            )

            y_usm = dpctl.tensor.asarray(
                ys,
                sycl_queue=self.sycl_queue
            )

            # Convert to dpnp
            x_dp = mytorch.Tensor(dp.asarray(x_usm))
            y_dp = mytorch.Tensor(dp.asarray(y_usm))
            self.queue.put((x_dp, y_dp))
        self.queue.put(self.stop_signal)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        batches = [
            indices[i:i+self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        for b in batches:
            self.index_queue.put(b)
        for _ in range(self.num_workers):
            self.index_queue.put(None)

        self.workers = []
        for _ in range(self.num_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self.index_queue,),
                daemon=True
            )
            t.start()
            self.workers.append(t)

        self.finished_workers = 0
        return self

    def __next__(self):
        item = self.queue.get() # blocks

        if item is self.stop_signal:
            self.finished_workers += 1
            if self.finished_workers == self.num_workers:
                raise StopIteration
            return self.__next__()

        return item