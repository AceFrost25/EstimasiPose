import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
from lib.knn.knn_pytorch import knn_pytorch

class KNearestNeighbor(Function):
    @staticmethod
    def forward(ctx, ref, query):
        ref = ref.float().cuda()
        query = query.float().cuda()
        inds = torch.empty(query.shape[0], 1, query.shape[2]).long().cuda()
        knn_pytorch.knn(ref, query, inds)  
        ctx.save_for_backward(inds)  # Save for backward pass (not used here)
        return inds

    @staticmethod
    def backward(ctx, grad_output):
        inds, = ctx.saved_tensors  # Not needed for this simple KNN, but a good practice
        return None, None

class TestKNearestNeighbor(unittest.TestCase):
    def test_forward(self):
        knn = KNearestNeighbor.apply  # Get the apply function for the forward pass
        D, N, M = 128, 100, 1000
        ref = (torch.rand(2, D, N)).cuda() 
        query = (torch.rand(2, D, M)).cuda()

        inds = knn(ref, query) 

        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())

        print(inds)

if __name__ == '__main__':
    unittest.main()

