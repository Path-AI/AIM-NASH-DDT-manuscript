def largest_hif(df, hif):
    if len(df)>1:
        return df[df[hif]==df[hif].max()]
    else:
        return df