"""
Library:
http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

Algorithm:
http://www.cs.t-kougei.ac.jp/SSys/Apriori.htm
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

dataset = [
    ['HOTEL', 'HOTEL_ITEM', "HOTEL_ITEM_CARD", "POINT", "MEMBER"],
    ['HOTEL', 'HOTEL_ITEM', 'HOTEL_ITEM', "HOTEL_OPTION", "HOTEL_OPTION", "HOTEL_ITEM_ONSITE", "HOTEL_OPTION_ONSITE", "HOTEL_OPTION_ONSITE", "HOTEL_OPTION_ONSITE", "NONMEMBER"],
    ['HOTEL', 'HOTEL_ITEM', "HOTEL_ITEM_ONSITE", "NONMEMBER"],
    ['AIR', 'AIR_ITEM', 'AIR_ITEM', 'AIR_ITEM_CARD'],
    ['AIR', 'AIR_ITEM', 'AIR_ITEM', 'AIR_ITEM', 'AIR_ITEM', 'AIR_ITEM_CARD']
]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(f'df:\n{df}')

frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(f'frequent_itemsets:\n{frequent_itemsets}')

"""
   support             itemsets  length
0      0.6              (HOTEL)       1
1      0.6         (HOTEL_ITEM)       1
2      0.6  (HOTEL_ITEM, HOTEL)       2
"""
