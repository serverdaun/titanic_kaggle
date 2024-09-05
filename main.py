import dill
import pandas as pd

with open('model/titanic_classifier_model.pkl', 'rb') as f:
    model = dill.load(f)

test_df = pd.read_csv('data/test.csv')

predictions = model['model'].predict(test_df)

prediction_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

output_file_path = 'model/titanic_predictions.csv'
prediction_df.to_csv(output_file_path, index=False)

print(f'Predictions saved to {output_file_path}')
