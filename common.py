from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix


def print_prediction(predictions, y, predictions_test, y_test,file_out):

    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions)
    recl = recall_score(y, predictions)
    cm = confusion_matrix(y, predictions)

    acc2 = accuracy_score(y_test, predictions_test)
    prec2 = precision_score(y_test, predictions_test)
    recl2 = recall_score(y_test, predictions_test)
    cm2 = confusion_matrix(y_test, predictions_test)

    '''
    print ('Confusion matrix TRAIN')
    print(cm)
    print('\nConfusion matrix DEV')
    print(cm2)
    print ('\n\t' + 'TRAIN\t |\tDEV')
    print('ACC:    ', round(acc,4), '\t|\t', round(acc2,4),
          '\nPREC:   ', round(prec,4),'\t|\t', round(prec2,4),
          '\nRECALL: ', round(recl,4),'\t|\t', round(recl2,4))
    '''
    
    file_out.write('Confusion matrix TRAIN\n')
    file_out.write(str(cm))
    file_out.write('\nConfusion matrix DEV\n')
    file_out.write(str(cm2))
    file_out.write('\n\t' + 'TRAIN\t |\tDEV\n')
    file_out.write('ACC:    '+ str(round(acc,4))+'\t|\t'+str(round(acc2,4))+\
          '\nPREC:   '+str(round(prec,4))+'\t|\t'+str(round(prec2,4))+\
          '\nRECALL: '+str(round(recl,4))+'\t|\t'+str(round(recl2,4)))
    file_out.write('\n\n\n')

