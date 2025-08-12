import cv2
import numpy as np
import time

def test_camera_basic():
    """Basic camera test without any ML"""
    print("🔍 Basic Camera Test")
    print("=" * 30)
    
    # Test different camera indices
    for i in range(3):
        print(f"\n📹 Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✅ Camera {i} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Frame read successfully: {frame.shape}")
                
                # Try to display the frame
                try:
                    cv2.imshow(f'Camera Test {i}', frame)
                    print(f"✅ Window created for camera {i}")
                    print(f"📋 Press any key to continue...")
                    
                    # Wait for key press
                    key = cv2.waitKey(0)
                    print(f"✅ Key pressed: {key}")
                    
                    cv2.destroyAllWindows()
                    cap.release()
                    
                    print(f"✅ Camera {i} test PASSED!")
                    return i  # Return working camera index
                    
                except Exception as e:
                    print(f"❌ Window display error: {e}")
            else:
                print(f"❌ Cannot read frame from camera {i}")
        else:
            print(f"❌ Cannot open camera {i}")
        
        cap.release()
    
    print("\n❌ No working camera found")
    return None

def test_opencv_display():
    """Test if OpenCV can display windows at all"""
    print("\n🖼️  Testing OpenCV Window Display")
    print("=" * 35)
    
    try:
        # Create a simple test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # Fill with color
        
        # Add text
        cv2.putText(test_image, "OpenCV Test Window", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(test_image, "Press any key to close", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Try to display
        cv2.imshow('OpenCV Display Test', test_image)
        print("✅ Test window created")
        print("📋 You should see a colored window with text")
        print("📋 Press any key in the window to continue...")
        
        key = cv2.waitKey(0)
        print(f"✅ Key received: {key}")
        
        cv2.destroyAllWindows()
        print("✅ OpenCV display test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV display error: {e}")
        return False

def test_camera_with_backends():
    """Test camera with different backends"""
    print("\n🔧 Testing Camera Backends")
    print("=" * 30)
    
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default")
    ]
    
    for backend, name in backends:
        print(f"\n📹 Testing {name}...")
        try:
            cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ {name} works! Frame: {frame.shape}")
                    
                    # Try to show frame
                    cv2.imshow(f'{name} Test', frame)
                    print(f"📋 {name} window should be visible - press any key")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cap.release()
                    
                    return backend, name
                else:
                    print(f"❌ {name} can't read frames")
            else:
                print(f"❌ {name} can't open camera")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ {name} error: {e}")
    
    return None, None

def main():
    print("🎥 Camera Diagnostic Tool")
    print("=" * 40)
    print("This will help us figure out why the camera window isn't opening")
    
    # Test 1: OpenCV display capability
    if not test_opencv_display():
        print("\n❌ PROBLEM: OpenCV cannot create windows")
        print("💡 SOLUTION: Try running as administrator")
        input("Press Enter to exit...")
        return
    
    # Test 2: Basic camera access
    working_camera = test_camera_basic()
    if working_camera is None:
        print("\n❌ PROBLEM: No camera access")
        print("💡 SOLUTIONS:")
        print("   - Check if other apps are using the camera")
        print("   - Check Windows camera privacy settings")
        print("   - Try running as administrator")
        input("Press Enter to exit...")
        return
    
    # Test 3: Camera backends
    working_backend, backend_name = test_camera_with_backends()
    if working_backend is None:
        print("\n❌ PROBLEM: Camera backends not working")
        input("Press Enter to exit...")
        return
    
    print(f"\n✅ SUCCESS!")
    print(f"   Working camera: {working_camera}")
    print(f"   Working backend: {backend_name}")
    print(f"   OpenCV display: Working")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Use: cv2.VideoCapture({working_camera}, {working_backend})")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()