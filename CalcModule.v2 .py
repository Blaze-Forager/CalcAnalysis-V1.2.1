"""
Advanced Calculator with Camera OCR Integration
----------------------------------------------
Features:
- Camera OCR for reading mathematical expressions
- Symbolic differentiation and integration
- Limit computation and Taylor series
- Interactive mode with live camera feed

Dependencies: sympy, opencv-python, pytesseract, pillow, numpy

Install:
    pip install sympy opencv-python pytesseract pillow numpy
    
For Tesseract OCR:
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - macOS: brew install tesseract
    - Linux: sudo apt install tesseract-ocr
"""

import sympy as sp
import cv2
import pytesseract
import numpy as np
from PIL import Image
import time
import re

class CalculusCalculator:
    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z')

    def differentiate(self, expr, var=None, order=1):
        if var is None:
            var = self.x
        return sp.diff(expr, var, order)

    def integrate(self, expr, var=None, lower=None, upper=None):
        if var is None:
            var = self.x
        if lower is not None and upper is not None:
            return sp.integrate(expr, (var, lower, upper))
        return sp.integrate(expr, var)

    def limit(self, expr, var=None, point=0, direction="+"):
        if var is None:
            var = self.x
        return sp.limit(expr, var, point, dir=direction)

    def taylor(self, expr, var=None, point=0, order=6):
        if var is None:
            var = self.x
        return sp.series(expr, var, point, order).removeO()

    def evaluate(self, expr, substitutions):
        return expr.subs(substitutions).evalf()

class CameraOCR:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.detected_text = ""
        
    def preprocess_image(self, frame):
        """Preprocess frame for better OCR accuracy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return processed
    
    def clean_math_expression(self, text):
        """Clean and format OCR text for mathematical expressions"""
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Common OCR corrections
        replacements = {
            'x': 'x',
            'X': 'x',
            '×': '*',
            '÷': '/',
            '^': '**',
            '²': '**2',
            '³': '**3',
            'π': 'pi',
            'e': 'E',
            ' ': '',  # Remove spaces
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Add multiplication signs where needed (e.g., 2x -> 2*x)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)
        text = re.sub(r'\)(\d)', r')*\1', text)
        text = re.sub(r'(\d)\(', r'\1*(', text)
        
        return text
    
    def perform_ocr(self, frame):
        """Perform OCR on preprocessed frame"""
        try:
            processed = self.preprocess_image(frame)
            pil_image = Image.fromarray(processed)
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            return text.strip()
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def draw_ui(self, frame, text, status=""):
        """Draw UI elements on frame"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (frame.shape[1]-10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 180), (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "CAMERA OCR CALCULATOR", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE: Capture | Q: Quit | P: Toggle View", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detected text
        cv2.putText(frame, "Detected:", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        lines = text.split('\n')[:2]
        y_offset = 115
        for line in lines:
            if line.strip():
                cv2.putText(frame, line[:60], (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # Status message
        if status:
            cv2.putText(frame, status, (20, 165), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def capture_expression(self):
        """Run camera and capture mathematical expression"""
        print("\n=== CAMERA OCR MODE ===")
        print("Position mathematical expression in front of camera")
        print("Press SPACE to capture, Q to quit, P to toggle preprocessing view")
        print("-" * 60)
        
        if not self.camera.isOpened():
            print("Error: Cannot open camera")
            return None
        
        show_preprocessed = False
        captured_expression = None
        status = "Ready to capture..."
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break
                
                # Choose display mode
                if show_preprocessed:
                    display_frame = cv2.cvtColor(
                        self.preprocess_image(frame), 
                        cv2.COLOR_GRAY2BGR
                    )
                else:
                    display_frame = frame.copy()
                
                # Perform OCR continuously for preview
                self.detected_text = self.perform_ocr(frame)
                
                # Draw UI
                display_frame = self.draw_ui(display_frame, self.detected_text, status)
                
                # Show frame
                cv2.imshow('Camera OCR Calculator', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space bar to capture
                    cleaned = self.clean_math_expression(self.detected_text)
                    captured_expression = cleaned
                    status = f"Captured: {cleaned}"
                    print(f"\n[CAPTURED] Raw: {self.detected_text}")
                    print(f"[CLEANED] Expression: {cleaned}")
                    time.sleep(1)  # Show confirmation
                    break
                elif key == ord('p'):
                    show_preprocessed = not show_preprocessed
                    mode = "Preprocessed" if show_preprocessed else "Normal"
                    status = f"View: {mode}"
        
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
        
        return captured_expression
    
    def close(self):
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

def process_expression(calc, expr_str):
    """Process and display calculus operations on expression"""
    x, y, z = calc.x, calc.y, calc.z
    
    try:
        expr = sp.sympify(expr_str)
        print(f"\n{'='*60}")
        print(f"Expression: {expr}")
        print(f"{'='*60}")
        
        # Derivative
        try:
            deriv = calc.differentiate(expr)
            print(f"1st Derivative (dx): {deriv}")
        except Exception as e:
            print(f"1st Derivative: Cannot compute ({e})")
        
        # Second derivative
        try:
            deriv2 = calc.differentiate(expr, order=2)
            print(f"2nd Derivative (d²x): {deriv2}")
        except Exception as e:
            print(f"2nd Derivative: Cannot compute ({e})")
        
        # Indefinite integral
        try:
            indef = calc.integrate(expr)
            print(f"Indefinite Integral: {indef} + C")
        except Exception as e:
            print(f"Indefinite Integral: Cannot compute ({e})")
        
        # Definite integral
        try:
            defin = calc.integrate(expr, x, 0, 1)
            print(f"Definite Integral [0,1]: {defin}")
        except Exception as e:
            print(f"Definite Integral [0,1]: Cannot compute ({e})")
        
        # Limit
        try:
            lim = calc.limit(expr, x, 0)
            print(f"Limit (x→0): {lim}")
        except Exception as e:
            print(f"Limit (x→0): Cannot compute ({e})")
        
        # Taylor series
        try:
            taylor = calc.taylor(expr, x, 0, 5)
            print(f"Taylor Series (5th order): {taylor}")
        except Exception as e:
            print(f"Taylor Series: Cannot compute ({e})")
        
        # Numeric evaluation
        try:
            numeric = calc.evaluate(expr, {x: 1})
            print(f"Value at x=1: {numeric}")
        except Exception as e:
            print(f"Value at x=1: Cannot compute ({e})")
        
        print(f"{'='*60}\n")
        return True
        
    except sp.SympifyError as e:
        print(f"\n[ERROR] Invalid expression syntax: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False

def main():
    print("\n" + "="*60)
    print(" ADVANCED CALCULATOR WITH CAMERA OCR")
    print("="*60)
    
    calc = CalculusCalculator()
    x, y, z = calc.x, calc.y, calc.z
    
    print("\n--- Input Format Guide ---")
    print("Variables: x, y, z")
    print("Operators: + - * / ** (power)")
    print("Functions: sin, cos, tan, exp, log, sqrt")
    print("Constants: pi, E")
    print("Examples: sin(x)**2 + 2*x, x**3 - 4*x + 1, exp(-x**2)")
    print("--------------------------\n")
    
    while True:
        print("\nChoose input method:")
        print("  1. Manual text input")
        print("  2. Camera OCR input")
        print("  3. Exit")
        
        choice = input("\nChoice (1/2/3): ").strip()
        
        if choice == '1':
            # Manual input
            try:
                expr_input = input("\nEnter expression: ").strip()
                if not expr_input:
                    continue
                process_expression(calc, expr_input)
            except KeyboardInterrupt:
                print("\n")
                continue
                
        elif choice == '2':
            # Camera OCR input
            try:
                ocr = CameraOCR(camera_index=0)
                captured = ocr.capture_expression()
                ocr.close()
                
                if captured:
                    print(f"\nProcessing captured expression: {captured}")
                    
                    # Allow user to edit before processing
                    edit = input("\nEdit expression? (y/n): ").strip().lower()
                    if edit == 'y':
                        captured = input("Enter corrected expression: ").strip()
                    
                    if captured:
                        process_expression(calc, captured)
                else:
                    print("\n[CANCELLED] No expression captured")
                    
            except KeyboardInterrupt:
                print("\n[CANCELLED] Camera capture interrupted")
                if 'ocr' in locals():
                    ocr.close()
            except Exception as e:
                print(f"\n[ERROR] Camera error: {e}")
                if 'ocr' in locals():
                    ocr.close()
                    
        elif choice == '3':
            print("\nExiting calculator. Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")