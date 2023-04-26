def draw_roc_curve(model, X, y):
    """
    Draws ROC curve of the statistical model.
    Source: https://www.statology.org/plot-roc-curve-python/

    Parameters
    ----------
    model : object
        Any classifier model from sklearn family
    
    X : pandas dataframce
        Matrix of features
        
    y : pandas series
        Vector of targets
    """ 
    
    
    from sklearn import metrics
    import matplotlib.pyplot as plt
    
    #define metrics
    y_pred_proba = model.predict_proba(X)[::,1]
    
    fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)
    auc = metrics.roc_auc_score(y, y_pred_proba)

    gini = 2*auc - 1

    #create ROC curve
    plt.plot(fpr,tpr,label=["AUC="+str(auc)[:6],"GINI="+str(gini)[:6]])
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    

def draw_precision_recall_area(model, X, y):
    """
    Draws Precision vs. Recall area of the statistical model.
    Source: https://www.datacamp.com/tutorial/precision-recall-curve-tutorial

    Parameters
    ----------
    model : object
        Any classifier model from sklearn family
    
    X : pandas dataframce
        Matrix of features
        
    y : pandas series
        Vector of targets
    """     
    
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    # Predict probability
    y_pred_proba = model.predict_proba(X)[::,1]

    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
    plt.fill_between(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Precision-Recall curve");

    
def draw_precision_recall_curves(model, X, y):
    """
    Draws Precision and Recall curves of the statistical model.
    Source: https://www.geeksforgeeks.org/data-normalization-with-pandas/

    Parameters
    ----------
    model : object
        Any classifier model from sklearn family
    
    X : pandas dataframce
        Matrix of features
        
    y : pandas series
        Vector of targets
    """     
    
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    y_pred_proba = model.predict_proba(X)[::,1]

    precision, recall, threshold = precision_recall_curve(y, y_pred_proba)

    # Plot the output.
    plt.plot(threshold, precision[:-1], c ='r', label ='PRECISION')
    plt.plot(threshold, recall[:-1], c ='b', label ='RECALL')
    plt.grid()
    plt.legend()
    plt.title('Precision-Recall Curve')

    
def draw_confusion_matrix(model, X, y, threshold=0.5 , labels=[False,True]):
    """
    Draws Confusion matrix of the statistical model.
    Source: https://www.w3schools.com/python/python_ml_confusion_matrix.asp

    Parameters
    ----------
    model : object
        Any classifier model from sklearn family
    
    X : pandas dataframce
        Matrix of features
        
    y : pandas series
        Vector of targets
        
    labels : list, optional
        specifying the labels values to be shown in the matrix, default is False & True
    """         
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    #y_model = model.predict(X)
    y_pred_proba = (model.predict_proba(X)[::, 1] > threshold).astype('float')

    confusion_matrix = confusion_matrix(y, y_pred_proba)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)

    cm_display.plot()
    plt.show()
    
    
def z_score_pandas_df(df, excl_col = []):
    """
    Applies z-score transformation (0 mean, 1 std deviation) to numerical columns of pandas dataframe.
    Source: https://www.w3schools.com/python/python_ml_confusion_matrix.asp

    Parameters
    ----------
    df : pd.DataFrame
        Input pandas dataframe
    
    excl_col : list of string
        List of column names that should be excluded from transformation


    Returns
    -------
    pd.DataFrame
        Pandas DataFrame(s) with the transformed columns
    
    """         
    df_z_scaled = df.copy()
    
    for column in df_z_scaled.columns:
        # https://www.skytowner.com/explore/checking_if_column_is_numeric_in_pandas_dataframe
        if df[column].dtype.kind in 'iufc' and column not in excl_col: # If you want the boolean type to be considered as in integer, include the character b as well
            df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()    
 
    return df_z_scaled


def sqllite_query(db, sql, bln_commit = True, bln_read_results_to_df = False):
    """
    Function executes a query to sqllite database and if specified returns the result to Pandas DataFrame

    Parameters
    ----------
    db : string
        string containing path to SQLLITE database

    sql : string
        string containing sql query to be executed

    bln_commit : boolean, optional
        boolean specified, whether the query should be committed

    bln_read_results_to_df : boolean, optional
        boolean specified, whether the query output should be outputed to Pandas DataFrame

    Returns
    -------
    pd.DataFrame
        Returns Pandas DataFrame with query output if specified in bln_read_results_to_df parameter
    """  
    
    import sqlite3
    import pandas as pd
    
    #Connecting to sqlite
    conn = sqlite3.connect(db, timeout = 5)

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    #Retrieving data
    cursor.execute(sql)
 
    #output = cursor.fetchall()

    if bln_commit:
        conn.commit()
    
    if bln_read_results_to_df:
        df = pd.read_sql_query(sql, conn) 
        
        df.head(10)
    
        #Closing the connection
        conn.close()
        
        return df  
        
    #Closing the connection
    conn.close()
    
    
def f_irr(nper, pmt, pv):
    """
    Returns the interest rate based on number of payments, payment amount and present value
    It's the same function as RATE in excel
    https://www.ablebits.com/office-addins-blog/excel-rate-function-calculate-interest-rate/

    Parameters
    ----------
    nper : integer
        Number of compounding periods
    
    pmt : double
        Payment
        
    pv : double
       Present value
        
    Returns
    -------
    double
        interest rate per given period
    """ 
    from numpy_financial import irr #(nper, pmt, pv, fv, when='end', guess=None, tol=None, maxiter=100)

    try:
        
        list_inp = [-pv] + [pmt] * int(nper) 
        
        return float(irr(list_inp)*12)
    
    except:
        
        return None
    
    
def psi(col):
    """
    Returns PSI (Population Stability Index)
    Rule of thumb: https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html
    
    PSI < 0.1: no significant population change
    PSI < 0.2: moderate population change
    PSI >= 0.2: significant population change

    Parameters
    ----------
    col : pandas DF column
        Input pandas column for which the metric should be calculated
        
    Returns
    -------
    double
        PSI statistics
    """     
    return ((col-col.mean()) * np.log(col/col.mean())).sum()


def draw_lift_curve(y_val, y_pred, step=0.01):
    """
    This function draws a cumulative lift curve
    SOURCE: https://howtolearnmachinelearning.com/code-snippets/lift-curve-code-snippet/
    """    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    #Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    #Create a real and predicted column for our new DataFrame and assign values
    aux_lift['real'] = y_val
    aux_lift['predicted'] = y_pred
    #Order the values for the predicted probability column:
    aux_lift.sort_values('predicted',ascending=False,inplace=True)
    
    #Create the values that will go into the X axis of our plot
    x_val = np.arange(step,1+step,step)
    #Calculate the ratio of ones in our data
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    #Create an empty vector with the values that will go on the Y axis our our plot
    y_v = []
    
    #Calculate for each x value its correspondent y value
    for x in x_val:
        num_data = int(np.ceil(x*len(aux_lift))) #The ceil function returns the closest integer bigger than our number 
        data_here = aux_lift.iloc[:num_data,:]   # ie. np.ceil(1.4) = 2
        ratio_ones_here = data_here['real'].sum()/len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)
           
   #Plot the figure
    fig, axis = plt.subplots()
    fig.figsize = (40,40)
    axis.plot(x_val, y_v, 'g-', linewidth = 3, markersize = 5)
    axis.plot(x_val, np.ones(len(x_val)), 'k-')
    axis.set_xlabel('Proportion of sample')
    axis.set_ylabel('Lift')
    plt.title('Lift Curve')
    plt.show()

    
def draw_score_vs_target_bins(score, target,  bins = 20, savepath = None, filename='calibration.png'):
    """   
    This function plots the default rate vs 1-average score in bin categories
    
    SOURCE: Home Credit Scoring Workflow
    """
    
    bins = np.percentile(score, np.linspace(0,100, bins + 1))

    scores = []
    brs = []
    for b in zip(bins[:-1], bins[1:]):
        scores += [score[(score>=b[0]) & (score<b[1])].mean()]
        brs += [target[(score>=b[0]) & (score<b[1])].mean()]


    plt.scatter(brs, scores)
    upperlimit = np.nanmax(scores + brs)
    plt.xlim([0, upperlimit])
    plt.ylim([0, upperlimit])
    plt.plot(np.linspace(0, max(scores + brs), 1000), np.linspace(0, max(scores + brs), 1000) , color='red')    
    plt.grid()
    plt.xlabel('default rate')
    plt.ylabel('1 - average score')
    if savepath is not None:
        plt.savefig(savepath + filename, bbox_inches='tight', dpi = 72)
    plt.show()    