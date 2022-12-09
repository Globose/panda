
    df_fem  = df[df['Sex']==0]
    df_male  = df[df['Sex']==1]
    
    print(df_fem['Survived'].value_counts())
    print(df_male['Survived'].value_counts())
    
    print(df_fem.describe())
    print(df_male.describe())