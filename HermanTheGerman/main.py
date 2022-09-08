from BEM import BEM

test = BEM()
test.set_constants(8, 2.61, 31, 3, 1.225)
print(f"Test results with c=0.5m: \n {test.solve(24.5, 0.5, 0.5, 0.01)}")
print(f"Test results with c=0.5m: \n {test.solve(24.5, 3, 0.5, 0.01)}")