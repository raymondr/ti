class KnightlyClassifier:
    """
    Python implementation of the Knightly embedded C classifier.
    Determines Target Type (Adult/Child/Dog) and Posture (Walking/Sitting/Laying)
    based on spatial coordinates and velocity.
    """
    def __init__(self):
        # Z-Height thresholds for Target Type (meters)
        self.threshold_dog_max_z = 0.35     # Anything under 35cm is likely a pet
        self.threshold_child_max_z = 0.85   # Between 35cm and 85cm is a child
                                            # Above 85cm is an adult
        
        # Velocity thresholds for Posture
        self.threshold_walking_min_v = 0.20 # >0.2m/s indicates active traversal
        
        # Z-Height thresholds for Posture when Stationary (Velocity < 0.2 m/s)
        self.threshold_laying_max_z = 0.40  # Roughly mattress height
        self.threshold_sitting_min_z = 0.45 # Roughly chair/edge-of-bed height
        self.threshold_sitting_max_z = 0.80 # Upper limit for a seated child

    def classify_target(self, x: float, y: float, z: float, velocity: float, vy: float = 0.0) -> dict:
        """
        Runs the decision tree on a single frame's tracked coordinates.
        
        :param x: Target X coordinate (meters)
        :param y: Target Y coordinate (meters)
        :param z: Target Z coordinate (meters - height from floor)
        :param velocity: Absolute velocity magnitude (m/s)
        :param vy: Velocity along the Y-axis (optional, used for door approach check)
        :return: Dictionary containing the classified type and posture
        """
        
        # ==========================================
        # 1. CLASSIFY TARGET TYPE (Adult/Child/Dog)
        # ==========================================
        if z <= self.threshold_dog_max_z:
            target_type = "DOG"
        elif z <= self.threshold_child_max_z:
            target_type = "CHILD"
        else:
            target_type = "ADULT"
            
        # ==========================================
        # 2. CLASSIFY POSTURE (Walking/Sitting/Laying)
        # ==========================================
        if velocity >= self.threshold_walking_min_v:
            # Target is actively moving
            posture = "WALKING"
        else:
            # Target is stationary (Velocity < 0.2m/s)
            if z <= self.threshold_laying_max_z:
                posture = "LAYING"
            elif self.threshold_sitting_min_z <= z <= self.threshold_sitting_max_z:
                posture = "SITTING"
            else:
                # Standing perfectly still (Velocity is low, but Z is tall)
                posture = "STANDING"
                
        # ==========================================
        # 3. DOOR APPROACH LOGIC
        # ==========================================
        # Assuming the sensor is mounted on the door at Y=0.
        # If the target is getting close (Y < 1.0m) while moving towards the door (vy < 0)
        is_approaching_door = bool(y < 1.0 and vy < -0.05)

        return {
            "type": target_type,
            "posture": posture,
            "approaching_door": is_approaching_door
        }

# --- Example Usage ---
if __name__ == "__main__":
    classifier = KnightlyClassifier()
    
    # Simulate a child sitting on the floor/bed
    result1 = classifier.classify_target(x=0.5, y=2.0, z=0.55, velocity=0.05)
    print(f"Test 1: {result1}")
    
    # Simulate an adult walking quickly towards the door
    result2 = classifier.classify_target(x=0.1, y=0.8, z=1.1, velocity=0.6, vy=-0.4)
    print(f"Test 2: {result2}")
