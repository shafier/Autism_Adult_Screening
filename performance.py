
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

def cross_value_score_trainSet_testSet(label, classifier, XtrainSet, ytrainSet, XtestSet, ytestSet, kFold):

    for name, clf in zip(label, classifier):
    
        scoresTrainSet = cross_val_score(
            estimator = clf, 
            X = XtrainSet,
            y = ytrainSet, 
            scoring = 'accuracy', 
            cv = kFold)
        
    
        print( "For {0}-fold cross-validation, Accuracy of Training Set: Mean = {1:.6f}, std = {2:.5f} [ {3} is used. ]".format( 
            kFold,
            scoresTrainSet.mean(), 
            scoresTrainSet.std(), 
            name ) )
    
        scoresTestSet = cross_val_score(
            estimator = clf, 
            X = XtestSet, 
            y = ytestSet, 
            scoring = 'accuracy', 
            cv = kFold)
    
    
        print( "For {0}-fold cross-validation, Accuracy of Test Set: Mean = {1:.6f}, std = {2:.5f} [ {3} is used. ] \n ".format( 
            kFold,
            scoresTestSet.mean(), 
            scoresTestSet.std(), 
            name ) )
            

def f1_score_trainSet_testSet(label, classifier, XtrainSet, ytrainSet, XtestSet, ytestSet):

    for name, clf in zip(label, classifier):
        predict_y_train_set = clf.predict( XtrainSet )
        predict_y_test_set = clf.predict( XtestSet )
        
        clsF1ScoreTrainSet = f1_score(ytrainSet, predict_y_train_set)
        print( "F1 Score for Training Set: {0: 6f} [ {1} is used. ] ".format( 
            clsF1ScoreTrainSet, name ) 
             )
        
        clsF1ScoreTestSet = f1_score(ytestSet, predict_y_test_set)
        print( "F1 Score for Test Set: {0: 6f} [ {1} is used. ] \n".format( 
            clsF1ScoreTestSet, name ) 
             )
        
