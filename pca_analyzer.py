#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copy import deepcopy
import seaborn as sns


class PcaAnalyzer:
    """
    Analyze and graph the PCA components
    Inputs:
        data: (df) preprocessed DataFrame
        subset: (list) of features pre-PCA (best to pass in SUBSET at top of the file)
        log_cols: (list) columns to log (best to pass in LOG_COLS at top of the file or have already logged columns in SUBSET)
        Scaler: type of scaler to use, such as StandardScaler, QuantileScaler, or RobustScaler
        pca_pct: (float) percentage of variance that must be explained by the PCA components
        
    Output:
        None: Call pca_explainer, pca_grapher, and get_select_components_df methods as needed
    Example:
        pca = PcaAnalyzer(data=df,subset=SUBSET,log_cols=LOG_COLS,Scaler=StandardScaler,pca_pct=.8)
        pca.pca_explainer()
        pca.pca_grapher(pca_indices=[1,2,3])
        pca.get_select_components_df(pca_indices=[1,2])

    """
    #XXX changing input to take scaler object, not class type
    def __init__(self,data,subset=None,log_cols=[],Scaler=StandardScaler(),pca_pct=None): 
        self.data = deepcopy(data)#.drop_duplicates() #XXX removing drop_duplicates. Causes issues since I am bringing in features without identifiers
        if not subset: #XXX changed default
            self.subset = list(self.data.columns)
        else:
            self.subset = subset #if len(subset) > 0 else self.get_nums(subset)
        self.type_checker()
        self.log_cols = log_cols
        self.scaler = Scaler
        self.pca_pct=pca_pct #XXX changed here too, so that not forced to project initially and matches PCA default
        self.is_scaled, self.is_imputed, self.is_logged = False, False, False
        self.pca = self.get_pca()

    def type_checker(self): # If have additional ones, consider moving these to a utils.py file
        from pandas.api.types import is_numeric_dtype
        non_nums=set()
        for col in set(self.data[self.subset].columns):
            if not is_numeric_dtype(self.data[col]):
                non_nums.add(col)
        if len(non_nums)>0:
            raise ValueError(f"{non_nums} is/are not numeric. Please pass in a DataFrame of numeric types")



    def log_transform(self):
        for col in self.log_cols:
            self.data[col] = np.log1p(self.data[col])


    def scaled(self,X):
        """Returns a scaled version of features"""
        X_scaled=self.scaler.fit_transform(X)
        self.is_scaled=True
        return X_scaled


    def imputed(self,X):
      
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        X_imp=imp.fit_transform(X)
        return X_imp


    def get_pca(self):
        if self.log_cols: 
            self.log_transform()
        self.X = self.data[self.subset] # will use self.X later, add it as an attribute
        
        #XXX want to be able to pass through without scaling
        if self.scaler:
            X_scaled=self.scaled(self.X)
        else:
            X_scaled = self.X
        
        X_imputed = self.imputed(X_scaled)

        pca = PCA(n_components=self.pca_pct)

    
        pca_comps = pca.fit(X_imputed)
        self.N_pca = pca.n_components_
        print(f"There are {self.N_pca} components numbered 1 through {self.N_pca}")
        
        columns = ['PC'+str(i) for i in range(1,self.N_pca+1)]
        self.pca_df= pd.DataFrame(pca.fit_transform(X_imputed),columns=columns)
        
        #XXX adding explained_variance_ratio
        self.expl_var_ratio = pca.explained_variance_ratio_


        return pca_comps

    
    def pca_explainer(self):
        # Special thanks to https://www.reneshbedre.com/blog/principal-component-analysis.html#pca-loadings-plots
        # for code suggestions
        
        print(f"Proportion of variance explained by the {self.N_pca} PCA components (largest to smallest)")
        expl_var_ratio = self.pca.explained_variance_ratio_
        print(expl_var_ratio)
        

        print(f"Cumulative variance explained by the {self.N_pca} PCA components (largest to smallest)")
        print(np.cumsum(expl_var_ratio))

        loadings = self.pca.components_

        pc_list = ["PC"+str(i) for i in list(range(1, self.N_pca+1))]
        loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
        loadings_df['variable'] = self.X.columns.values
        loadings_df = loadings_df.set_index('variable')
        #XXX returning at end
        #print("PCA Loadings")
        #print(loadings_df)
        
        ###XXX changed figsize
        fig, ax = plt.subplots(figsize = (self.N_pca + 2,.4 * len(set(self.subset))))
        ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral', vmin = -1, vmax = 1, center = 0)
        plt.show()

        print("Scree plot")


        columns = ['PCA'+str(i+1) for i in range(self.N_pca)]

        df_cluster_mnt_totals = pd.DataFrame([expl_var_ratio], columns = columns)

        df_cluster_mnt_totals=df_cluster_mnt_totals.T.reset_index().rename({0:'Percent of Variance Explained'},axis=1)

        fig=df_cluster_mnt_totals.sort_values('Percent of Variance Explained', ascending=False).plot.bar(color='#4503fc')

        fig.set_xticklabels(columns)
        
        return loadings_df
        
    def scree_plot(self, component = None):
        # XXX Added separate scree plot
        PC_values = np.arange(self.N_pca) + 1
        plt.plot(PC_values, self.expl_var_ratio, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        if component:
            plt.axhline(y = self.expl_var_ratio[component-1], linestyle = '--', color = 'red')
        plt.show()
    
    #XXX returns total variance explained by first <n> components
    def total_var(self,n):
        return self.expl_var_ratio[:n].sum()
        
    

    def validate_indices(self, pca_indices):
        """Validate the set of passed indices"""
        if not pca_indices: 
            raise ValueError("Please pass at least one PCA component")
       

        try:
            pca_indices = sorted(set(pca_indices))
        except:
            raise ValueError("Did you pass a set of sortable indices?")
        else:
            for idx in pca_indices:
                if type(idx) is not int:
                    raise ValueError("Indices must be integers")

            max_idx, min_idx=max(pca_indices), min(pca_indices)

            if max_idx > self.N_pca or min_idx < 1:
                raise ValueError(f"Did you pass a set of indices between 1 and {self.N_pca}?")
        


    def get_select_components(self,pca_indices):
        
        """
        Returns the PCA components 
        """
        self.validate_indices(pca_indices)
        return tuple([self.pca_df['PC'+str(x)] for x in pca_indices])

    def get_select_components_df(self,pca_indices):
        """
        Returns select components as a DataFrame
        """
        pca_comps=self.get_select_components(pca_indices)
        return pd.DataFrame(pca_comps).transpose()
      
    def pca_grapher(self,pca_indices=[]):
        """
        Ex: my_pca.pca_grapher(pca_indices=[1,2,4])
        """

        if len(pca_indices) != 3:
            raise ValueError("Please pass exactly 3 indices for PCA dimensions")

        self.x, self.y, self.z=self.get_select_components(pca_indices)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.x,self.y,self.z, c="maroon", marker="o",alpha=0.2 )
        ax.set_title(f"A 3D Projection Of Data In The Reduced Dimension for indices {pca_indices}")
        #XXX added axis labels
        ax.set_xlabel(f'Principal Component:{pca_indices[0]}')
        ax.set_ylabel(f'Principal Component: {pca_indices[1]}')
        ax.set_zlabel(f'Principal Component: {pca_indices[2]}')
        plt.show()

 