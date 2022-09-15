from BEM import BEM

test = BEM()
test.set_constants(8, 2.61, 31, 3, 1.225)
c0 = 0.5
c1 = 1
c2 = 1.5
print(f"Test results with c={c0}m: \n {test.solve(24.5, c0, 0.5, 0.01)}")
print(f"Test results with c={c1}m: \n {test.solve(24.5, c1, 0.5, 0.01)}")
print(f"Test results with c={c2}m: \n {test.solve(24.5, c2, 0.5, 0.01)}")