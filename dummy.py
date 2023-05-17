import numpy as np
import pandas as pd
from scipy.stats import t, norm
from mad import mad

def f(err): 
  return np.sum(t.logpdf((resp - mu) / err, df=4) - np.log(err))

if (0):
  cnst = [1,2,3,4,5]
  resp = [1,2,3,4,6]

  mu = np.zeros(len(cnst))

  er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-32)

  p = np.exp(er_est)
  print(np.log(p))
  print(f(p))

technologies = {
    'Courses':["Spark","PySpark","Python","pandas"],
    'Fee' :[20000,25000,22000,30000],
    'Duration':['30days','40days','35days','50days'],
    'Discount':[1000,2300,1200,2000]}
df = pd.DataFrame(technologies)
print(df)


# Below are quick examples.
# Insert list into cell using df.at().
# df.at[1, 'Duration'] = {"h":1}#['30days', '35days']

print(df)

# # Insert list index into cell by df.iat() method.
# df.iat[1, df.columns.get_loc('Duration')] = ['30days', '35days']

# # Get list index into cell using df.loc() attribute.
df.loc[:, 'new'] = [ {"a":1}, {"b":2}, {"c":3}, {"d":4}]

print(df)




