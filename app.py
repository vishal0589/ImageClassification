import gradio as gr
from fastai.learner import load_learner
from fastai.vision.all import PILImage

# Load your trained model
learn = load_learner('export.pkl')

def predict(image):
    # Make a prediction
    pred,pred_idx,probs = learn.predict(image)
    # Return the prediction and probability
    return f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

# Define the Gradio interface
iface = gr.Interface(fn=predict, inputs='image', outputs='text')

# Run the interface
if __name__ == "__main__":
    iface.launch()
