import argparse
import pandas as pd
import sklearn.externals 
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def main(X_test_path, y_test_path, model_path):
    # Load the test data
    X_test = load_data(X_test_path)
    y_test = load_data(y_test_path)

    # Load the trained model
    model = joblib.load(model_path) 

    # Make predictions
    predictions = model.predict(X_test)

    # Compare predictions with actual values
    results = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': predictions})

    print(results)

    # Calculate accuracy or other metrics if needed
    accuracy = (results['Actual'] == results['Predicted']).mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML model predictions')
    parser.add_argument('--X_test', type=str, required=True, help='Path to X_test.csv')
    parser.add_argument('--y_test', type=str, required=True, help='Path to y_test.csv')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')

    args = parser.parse_args()
    main(args.X_test, args.y_test, args.model)
