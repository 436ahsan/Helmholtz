import helmholtz as hm


class TestWindowSvd:

    def test_window_svd_1d_helmholtz(self):
        n = 16
        kh = 0.5
        window_shape = (n,)
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        s, vh = old_code.repetitive.window_svd.get_window_svd(a, window_shape, num_sweeps=1000)

        # For sufficiently smooth test vectors, the SVD should reveal two large singular values (corresponding to the
        # sine and cosine null-space components / left- and right-travelling waves) and the rest are small.
        assert s[0] == 1
        assert s[1] > 0.3
        assert max(s[2:]) < 1e-5
