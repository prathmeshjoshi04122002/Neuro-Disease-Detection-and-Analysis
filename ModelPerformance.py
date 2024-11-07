predictions = model.predict(X_val_prep)
predictions = [1 if x>0.5 else 0 for x in predictions]
accuracy = accuracy_score(y_val, predictions) print('Val Accuracy = %.2f' %
accuracy)
confusion_mtx = confusion_matrix(y_val, predictions)
cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), norm alize=False)
Test Accuracy = 1.00
