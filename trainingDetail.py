import pandas as pd
import matplotlib.pyplot as plt

unpickled = pd.read_pickle('./models/history.pkl')
unpickled_df = pd.DataFrame(unpickled)
unpickled_df.plot.line()