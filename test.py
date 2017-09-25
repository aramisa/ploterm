import numpy as np
import ploterm
import time

a = list(np.random.rand(500).astype(float))
start = time.time()
g = ploterm.ascii_plot_simple_wrap(a, 100, 10)
end = time.time() - start
print g
print end
