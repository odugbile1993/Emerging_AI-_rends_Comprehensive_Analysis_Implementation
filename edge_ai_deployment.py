"""
Edge AI Deployment Script for Recyclable Classification
Simulates deployment on edge devices like Raspberry Pi
"""

import tensorflow as tf
import numpy as np
import cv2
import time
from PIL import Image
import os

class EdgeAIRecyclingClassifier:
    """
    Edge AI deployment class for real-time recyclable item classification
    """
    
    def __init__(self, model_path='recyclable_classifier.tflite'):
        """
        Initialize the Edge AI classifier
        
        Args:
            model_path: Path to the TensorFlow Lite model
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = ['paper', 'plastic', 'glass', 'metal', 'non-recyclable', 
                           'cardboard', 'electronics', 'organic', 'textiles', 'other']
        
        self.load_model()
    
    def load_model(self):
        """Load the TFLite model and allocate tensors"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("✅ TFLite model loaded successfully")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Input type: {self.input_details[0]['dtype']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model inference
        
        Args:
            image: numpy array or PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize to model input size (32x32 for this example)
        image = cv2.resize(image, (32, 32))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """
        Perform inference on input image
        
        Args:
            image: Input image for classification
            
        Returns:
            dict: Prediction results with class and confidence
        """
        if self.interpreter is None:
            return {"error": "Model not loaded"}
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get prediction results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        return {
            'class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'inference_time_ms': inference_time,
            'all_predictions': dict(zip(self.class_names, predictions))
        }
    
    def benchmark_performance(self, num_iterations=100):
        """
        Benchmark model performance for edge deployment
        
        Args:
            num_iterations: Number of inference iterations for benchmarking
        """
        print(f"\n🧪 Benchmarking Performance ({num_iterations} iterations)...")
        
        # Create dummy data
        dummy_image = np.random.rand(32, 32, 3).astype(np.float32)
        
        inference_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.predict(dummy_image)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_time = np.std(inference_times)
        
        print(f"✅ Average Inference Time: {avg_time:.2f}ms")
        print(f"✅ Minimum Inference Time: {min_time:.2f}ms")
        print(f"✅ Maximum Inference Time: {max_time:.2f}ms")
        print(f"✅ Standard Deviation: {std_time:.2f}ms")
        print(f"✅ FPS: {1000/avg_time:.1f} frames per second")
        
        return {
            'avg_inference_time': avg_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'std_inference_time': std_time,
            'fps': 1000/avg_time
        }

def simulate_real_time_detection():
    """Simulate real-time recyclable item detection"""
    print("\n🎯 Simulating Real-time Recycling Detection...")
    
    # Initialize classifier
    classifier = EdgeAIRecyclingClassifier()
    
    # Simulate processing multiple items
    test_items = [
        {'name': 'plastic_bottle', 'simulated_confidence': 0.85},
        {'name': 'glass_jars', 'simulated_confidence': 0.92},
        {'name': 'paper_cup', 'simulated_confidence': 0.78},
        {'name': 'metal_can', 'simulated_confidence': 0.88},
    ]
    
    print("\n🔄 Processing Recyclable Items...")
    for item in test_items:
        # Simulate prediction (in real scenario, this would use actual images)
        result = {
            'class': item['name'].split('_')[0],  # Extract material
            'confidence': item['simulated_confidence'],
            'inference_time_ms': np.random.uniform(5, 15)
        }
        
        print(f"📦 Item: {item['name']}")
        print(f"   🎯 Classification: {result['class']}")
        print(f"   📊 Confidence: {result['confidence']:.2f}")
        print(f"   ⚡ Inference Time: {result['inference_time_ms']:.2f}ms")
        
        # Decision making based on confidence
        if result['confidence'] > 0.8:
            print("   ✅ HIGH CONFIDENCE - Ready for sorting")
        else:
            print("   ⚠️  LOW CONFIDENCE - Requires manual review")
        print()

def main():
    """Main demonstration function"""
    print("="*60)
    print("♻️ EDGE AI RECYCLING CLASSIFIER - DEPLOYMENT DEMO")
    print("="*60)
    
    # Initialize and benchmark
    classifier = EdgeAIRecyclingClassifier()
    
    # Benchmark performance
    performance_stats = classifier.benchmark_performance(num_iterations=50)
    
    # Simulate real-time detection
    simulate_real_time_detection()
    
    # Edge AI Benefits Summary
    print("\n" + "="*50)
    print("🚀 EDGE AI DEPLOYMENT BENEFITS")
    print("="*50)
    print("✅ Ultra-low Latency: {:.2f}ms per inference".format(performance_stats['avg_inference_time']))
    print("✅ High Throughput: {:.1f} FPS".format(performance_stats['fps']))
    print("✅ Privacy: All processing happens locally")
    print("✅ Offline Operation: No internet required")
    print("✅ Low Power: Optimized for edge devices")
    print("✅ Real-time Decision: Instant sorting decisions")
    print("✅ Scalable: Can deploy across multiple sorting stations")
    
    print("\n🎯 Real-world Applications:")
    print("   • Smart recycling bins")
    print("   • Automated sorting facilities")
    print("   • Mobile recycling apps")
    print("   • Educational tools for waste management")

if __name__ == "__main__":
    main()
