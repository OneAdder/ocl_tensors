import pytest
import numpy as np
from ocl_tensors.tensors import tensor


class TestTensors:
    def test_vector_float64(self):
        a = np.array([1.5, 2.5, 3.5]).astype(np.float64)
        res = tensor(a)
        np.testing.assert_array_equal(res.to_cpu(), a)
        assert res.dtype == np.float64
        assert res.shape == a.shape
        assert res._shape_gpu == (len(a),)

    def test_vector_int32(self):
        a = np.array([1, 2, 3]).astype(np.int32)
        res = tensor(a)
        np.testing.assert_array_equal(res.to_cpu(), a)
        assert res.dtype == np.int32
        assert res.shape == a.shape
        assert res._shape_gpu == (len(a),)

    def test_2d_matrix(self):
        a = np.array([[1, 2, 3], [1, 2, 3]])
        res = tensor(a)
        np.testing.assert_array_equal(res.to_cpu(), a)
        assert res.dtype == np.int64
        assert res.shape == a.shape
        assert res._shape_gpu == (np.prod(a.shape),)

    def test_3d_matrix(self):
        a = np.array([[[1, 2, 3], [1, 2, 3]],
                      [[1, 2, 3], [1, 2, 3]]]).astype(np.int32)
        res = tensor(a)
        np.testing.assert_array_equal(res.to_cpu(), a)
        assert res.dtype == np.int32
        assert res.shape == a.shape
        assert res._shape_gpu == (np.prod(a.shape),)


class TestTensorSumOperations:
    @pytest.mark.parametrize('tens, add, gold', [
        (tensor([1, 2, 3]), 2, np.array([3, 4, 5])),
        (tensor([1, 2, 3], dtype=np.float64),
         2.5, np.array([3.5, 4.5, 5.5])),
        (tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]), 2,
         np.array([[[3, 4, 5], [3, 4, 5]], [[3, 4, 5], [3, 4, 5]]]))])
    def test_add_scalar(self, tens, add, gold):
        np.testing.assert_array_equal(tens.scalar_sum(add).to_cpu(), gold)

    @pytest.mark.parametrize('tens1, tens2, gold', [
        (tensor([1, 2, 3]), tensor([1, 2, 3]), np.array([2, 4, 6])),
        (tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
         tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
         np.array([[[2, 4, 6], [2, 4, 6]], [[2, 4, 6], [2, 4, 6]]]))])
    def test_tensor_sum(self, tens1, tens2, gold):
        np.testing.assert_array_equal(tens1.tensor_sum(tens2).to_cpu(), gold)


class TestTensorProductOperations:
    @pytest.mark.parametrize('tens, mult, gold', [
        (tensor([1, 2, 3]), 2, np.array([2, 4, 6])),
        (tensor([1, 2, 3], dtype=np.float64),
         2.5, np.array([2.5, 5, 7.5])),
        (tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]), 2,
         np.array([[[2, 4, 6], [2, 4, 6]], [[2, 4, 6], [2, 4, 6]]]))])
    def test_mult_scalar(self, tens, mult, gold):
        np.testing.assert_array_equal(tens.scalar_mult(mult).to_cpu(), gold)

    @pytest.mark.parametrize('tens1, tens2, gold', [
        (tensor([1, 2, 3]), tensor([1, 2, 3]), np.array([1, 4, 9])),
        (tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
         tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]),
         np.array([[[1, 4, 9], [1, 4, 9]], [[1, 4, 9], [1, 4, 9]]]))])
    def test_inner_product(self, tens1, tens2, gold):
        np.testing.assert_array_equal(tens1.inner_product(tens2).to_cpu(), gold)

    @pytest.mark.parametrize('tens1, tens2, gold', [
        (tensor([1, 2, 3]), tensor([1, 2, 3]),
         np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])),
        (tensor([1, 2, 3, 4]), tensor([1, 2, 3]),
         np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]])),
        (tensor([1, 2, 3]), tensor([1, 2, 3, 4]),
         np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]]))])
    def test_outer_product(self, tens1, tens2, gold):
        np.testing.assert_array_equal(tens1.outer_product(tens2).to_cpu(), gold)

    @pytest.mark.parametrize('tens1, tens2, gold', [
        (tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
         tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
         np.array([[6, 12, 18], [6, 12, 18], [6, 12, 18]])),
        (tensor([[1], [2], [3]]), tensor([[1, 2, 3, 4]]),
         np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]])),
        (tensor([[1, 2], [1, 2], [1, 2]]), tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
         np.array([[3, 6, 9, 12], [3, 6, 9, 12], [3, 6, 9, 12]]))])
    def test_matmul(self, tens1, tens2, gold):
        np.testing.assert_array_equal(tens1.matmul(tens2).to_cpu(), gold)


class TestMagicMethods:
    @pytest.mark.parametrize('tens, other, gold', [
        (tensor([1, 2, 3]), 1, np.array([2, 3, 4])),
        (tensor([1, 2, 3]), tensor([1, 2, 3]), np.array([2, 4, 6]))])
    def test_add(self, tens, other, gold):
        np.testing.assert_array_equal((tens + other).to_cpu(), gold)

    @pytest.mark.parametrize('tens, other, gold', [
        (tensor([1, 2, 3]), 2, np.array([2, 4, 6])),
        (tensor([1, 2, 3]), tensor([1, 2, 3]), np.array([1, 4, 9]))])
    def test_mul(self, tens, other, gold):
        np.testing.assert_array_equal((tens * other).to_cpu(), gold)

    @pytest.mark.parametrize('tens1, tens2, gold', [
        (tensor([1, 2, 3]), tensor([1, 2, 3]),
         np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])),
        (tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
         tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
         np.array([[6, 12, 18], [6, 12, 18], [6, 12, 18]]))])
    def test_pow(self, tens1, tens2, gold):
        np.testing.assert_array_equal((tens1 ** tens2).to_cpu(), gold)
