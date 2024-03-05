def init_processing(X):
    '''
    Initial processing, takes in train and test data.
    '''
    # transfrom y/n columns to 1/0
    X['international_plan'] = X['international_plan'].map({'yes':1,'no':0})
    X['voice_mail_plan'] = X['voice_mail_plan'].map({'yes':1,'no':0})
    # make 'state' uppercase to eliminate inconsitencies 
    X['state'] = X['state'].str.upper()
    # turn `phone_number` into int
    #X['phone_number'] = X['phone_number'].str.replace('-','').astype(int)
    return X


def Resampling(X,y):
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X,y)

    # Preview new class distribution
    print('Synthetic sample class distribution: \n')
    print(pd.Series(y_train_resampled).value_counts())
    return X_train_resampled, y_train_resampled


def select_features(X,selector):
    '''
    Given a dataframe and selector, use the selector
    to get the most important features.
    '''
    imp_feat = dict(zip(X.columns,selector.get_support()))
    selected_array = selector.transform(X)
    selected_df = pd.DataFrame(selected_array,columns=[col for col in X.columns if imp_feat[col]],
                              index = X.index)
    return selected_df


def scale_values(X, scaler):
    '''
    Takes DataFrame and fitted scaler as input.
    '''
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled,columns=X.columns,index=X.index)
    return X_scaled


class ModCrossVal():
    '''Create model and see the crossvalidation more easily'''

    def __init__(self, model, model_name, X, y, cv_now=True):
        self.model = model
        self.name = model_name
        self.X = X
        self.y = y
    
        # For CV results
        self.kfolds = None
        self.cv_results = None
        self.cv_mean = None
        self.cv_std = None
    
        if cv_now:
            self.cross__val()
    
    def cross__val(self,X=None,y=None, kfolds=10):
        '''
        Perform cross validation and return results.
    
        Args:
         X:
          Optional; Training data to perform CV on. Otherwise use X from object
         y:
          Optional; Training data to perform CV on. Otherwise use y from object
         kfolds:
          Optional; Number of folds for CV (default is 8)  
        '''
    
        cv_X = X if X else self.X
        cv_y = y if y else self.y
        self.kfolds=kfolds
    
        self.cv_results = cross_validate(self.model,cv_X,cv_y,scoring='recall',return_train_score=True,cv=kfolds)
        self.cv_train_mean = np.mean(self.cv_results['train_score'])
        self.cv_test_mean = np.mean(self.cv_results['test_score'])

    def finetune_c(self):
        C_values = [0.0001, 0.001, 0.01, 0.1, 1]
    
    def cv_summary(self):
        
        summary = {
            'model_name':self.name,'kfolds':self.kfolds,'cv_train_mean':self.cv_train_mean,'cv_test_mean':self.cv_test_mean}
    
        cv_summary = pd.DataFrame(summary,columns=['model_name', 'kfolds','cv_train_mean','cv_test_mean'],
                                  index=range(1))
        return cv_summary