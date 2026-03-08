import pandas as pd
import rispy as rp

def doi_clean(df, doi_string):
    df['doi'] = df[doi_string].str.lower().str.strip()
    df['doi'] = df['doi'].str.replace('https://doi.org/', '', regex=False)
    return df

sc1_rp_path = 'scopus_bootstrap_1.ris'
sc2_path = 'scopus_bootstrap_2.csv'
sc_filter = 'scopus_bootstrap_filter.csv'

with open(sc1_rp_path, 'r', encoding='utf-8-sig') as file:
    entries = rp.load(file)


df = pd.DataFrame(entries)
df2 = pd.read_csv(sc2_path)
df_filter = pd.read_csv(sc_filter)
today = pd.read_csv('bootstrap_2312.csv')
print(df_filter.columns)
df = doi_clean(df, 'doi')
df2 = doi_clean(df2, 'DOI')
df_filter = doi_clean(df_filter, 'DOI')
df_today = doi_clean(today,'DOI')
df_filter = df_filter.sort_values(by='doi', ascending=True).reset_index()
df2 = df2.sort_values(by='doi', ascending=True).reset_index()
df_today = df_today.sort_values(by='doi', ascending=True).reset_index()
print(df.columns)
print(df2.columns)

print(f"DF has {df['doi'].shape[0]}")
print(f"DF2 has {df2['doi'].shape[0]}")
print(f"DF_filter has {df_filter['doi'].shape[0]}")

td_filter_commom = df_today[df_today['doi'].isin(df_filter['doi'])]
td_filter_not_commom = df_today[~df_today['doi'].isin(df_filter['doi'])]
commom = df_filter[df_filter['doi'].isin(df2['doi'])]
not_commom = df_filter[~df_filter['doi'].isin(df2['doi'])]
commom_filter = df2[df2['doi'].isin(df_filter['doi'])]
print(f"Commom has {commom.shape[0]}")
print(f"Not Commom has {not_commom.shape[0]}")
print(not_commom)