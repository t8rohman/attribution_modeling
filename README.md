# Tutorial on Attribution Modeling

Welcome to this tutorial on Attribution Modeling with Python! In this guide, we will explore how to utilize a Python module designed for conducting various Attribution Models. The module covers several types of Attribution Models, including First-Touch, Last-Touch, Linear, Time-Decay, and Position-Based. Each model offers a distinct approach to credit distribution, providing marketers with valuable insights into customer behavior and channel performance.

## Getting Started

Before we begin, ensure you have the following libraries installed:

- Pandas
- Numpy
- Matplotlib

For the specific versions of these libraries, please refer to the "requirements.txt" file.

## Dataset Format

To begin, your data should adhere to the following format:

<img src="https://github.com/t8rohman/attribution_modeling/blob/main/images/dataset-format.png" alt="GitHub Logo" width="1000">

The expected columns include:

- **ID:** Unique identifier for each customer's journey (e.g., cookie)
- **Time:** Timestamp to sequence and order customer journeys
- **Channel**: Channel where the customer interacted
- **Conversion:** Binary variable indicating whether a conversion occurred at that touchpoint

## Simple Attribution Modeling

The simple attribution modeling covers First-Touch, Last-Touch, Linear, Time-Decay, and Position-Based models. To use these, create a `SimpleModels` object from the `simple_models` module. You can then call individual attribution methods using the object's docstring. 

Alternatively, use the `combine_all_methods()` method to execute all methods at once.

```python
from attribution_modeling import simple_models

simple_att = simple_models.SimpleModels(df,
                                       col_id='cookie',
                                       col_time='time',
                                       col_chan='channel',
                                       col_conv='conversion')
simple_att.combine_all_methods()
```

## Markov Chain Modeling

For Markov Chain Modeling, data preprocessing is necessary using the `DataPreprocess` class from the `markov_chain` module. After preprocessing, call `preprocess_data()` and save the resulting dataframe as `df_paths` for subsequent analysis. 

The following code demonstrates the data preprocessing step:

```python
from attribution_modeling import markov_chain

markov_chain_preprocess = markov_chain.DataPreprocess(df,
                                                      col_id='cookie',
                                                      col_time='time',
                                                      col_chan='channel',
                                                      col_conv='conversion')
df_paths = markov_chain_preprocess.preprocess_data()
```

Once `df_paths` is prepared, move on to the analysis. Instantiate the `MarkovChainAnalysis` class from the `markov_chain` module, passing `df_paths` to the class. Before obtaining the final result, create a transition matrix using the `create_transition_matrix()` method. 

Finally, call `removal_attribution_channels()` on your MarkovChainAnalysis object to acquire the removal effects and channel attributions.

```python
from attribution_modeling import markov_chain

markov_chain_analysis = markov_chain.MarkovChainAnalysis(df_paths)
transition_matrix = markov_chain_analysis.create_transition_matrix()

att_markov = markov_chain_analysis.removal_attribution_channels()
```

## Extras!

In addition to the practical implementation, this repository also includes theoretical explanations of all attribution modeling methodologies. You can find this information in the "theory" folder if you're interested in understanding the underlying concepts.

## References

- Markov Chains explained visually by <a href='https://setosa.io/ev/markov-chains/'>Victor Powell</a>
- Multi-Touch Attribution Modelling with Markov Chains by <a href='https://medium.com/@aditya2590/multi-channel-attribution-modelling-with-markov-chains-fbf3bdab2ca8'>Aditya Saxena</a>
- Attribution Modeling using Markov Chain by <a href='https://medium.com/@akanksha.etc302/attribution-modeling-using-markov-chain-88fc6c0a499e'>Akanksha Anand (Ak)</a>
