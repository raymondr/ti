class KnightlyClassifier:
    """
    Python implementation of the Knightly embedded C classifier.
    Determines Target Type (Adult/Child/Dog) and Posture (Walking/Sitting/Laying)
    based on spatial coordinates, velocity, and bed location.
    """
    def __init__(self, bed_location=None):
        """
        :param bed_location: Bounding box for bed location. Can be:
                             - list/tuple of 4 floats: [min_x, max_x, min_y, max_y]
                             - list/tuple of 6 floats: [min_x, max_x, min_y, max_y, min_z, max_z]
                             - dict with keys (min_x/max_x/min_y/max_y or minX/maxX/minY/maxY)
                             - BoundingBox object with min_x, max_x, min_y, max_y attributes
        """
        # Z-Height thresholds for Target Type (meters)
        self.threshold_dog_max_z = 0.35     # Anything under 35cm is likely a pet
        self.threshold_child_max_z = 0.85   # Between 35cm and 85cm is a child
                                            # Above 85cm is an adult
        
        # Velocity thresholds for Posture
        self.threshold_walking_min_v = 0.50 # >0.5m/s indicates active traversal
        
        # Z-Height thresholds for Posture when Stationary (Velocity < 0.5 m/s)
        self.threshold_laying_max_z = 0.40  # Roughly mattress height
        self.threshold_sitting_min_z = 0.45 # Roughly chair/edge-of-bed height
        self.threshold_sitting_max_z = 0.80 # Upper limit for a seated child
        
        self.bed_location = bed_location

    def parse_bed_bounds(self, bed_location=None):
        """
        Parses bed_location into (min_x, max_x, min_y, max_y, min_z, max_z).
        Returns None if no bed is configured or invalid.
        """
        bed = bed_location if bed_location is not None else self.bed_location
        if bed is None:
            return None

        min_x, max_x, min_y, max_y = None, None, None, None
        min_z, max_z = None, None

        if isinstance(bed, str):
            bed = bed.replace(',', ' ').split()

        if isinstance(bed, (list, tuple)):
            items = []
            for item in bed:
                if isinstance(item, str):
                    items.extend(item.replace(',', ' ').split())
                else:
                    items.append(item)
            try:
                vals = [float(v) for v in items]
                if len(vals) >= 4:
                    min_x, max_x, min_y, max_y = vals[0], vals[1], vals[2], vals[3]
                if len(vals) >= 6:
                    min_z, max_z = vals[4], vals[5]
            except ValueError:
                return None
        elif isinstance(bed, dict):
            try:
                min_x = bed.get("min_x") or bed.get("minX") or bed.get("x_min")
                max_x = bed.get("max_x") or bed.get("maxX") or bed.get("x_max")
                min_y = bed.get("min_y") or bed.get("minY") or bed.get("y_min")
                max_y = bed.get("max_y") or bed.get("maxY") or bed.get("y_max")
                min_z = bed.get("min_z") or bed.get("minZ") or bed.get("z_min")
                max_z = bed.get("max_z") or bed.get("maxZ") or bed.get("z_max")
                if min_x is not None: min_x = float(min_x)
                if max_x is not None: max_x = float(max_x)
                if min_y is not None: min_y = float(min_y)
                if max_y is not None: max_y = float(max_y)
                if min_z is not None: min_z = float(min_z)
                if max_z is not None: max_z = float(max_z)
            except (ValueError, TypeError):
                return None
        elif hasattr(bed, "min_x"):
            try:
                min_x = float(getattr(bed, "min_x"))
                max_x = float(getattr(bed, "max_x"))
                min_y = float(getattr(bed, "min_y"))
                max_y = float(getattr(bed, "max_y"))
                min_z = getattr(bed, "min_z", None)
                max_z = getattr(bed, "max_z", None)
                if min_z is not None: min_z = float(min_z)
                if max_z is not None: max_z = float(max_z)
            except (ValueError, TypeError):
                return None

        if min_x is None or max_x is None or min_y is None or max_y is None:
            return None

        # Normalize min and max ordering defensively
        min_x, max_x = min(min_x, max_x), max(min_x, max_x)
        min_y, max_y = min(min_y, max_y), max(min_y, max_y)
        if min_z is not None and max_z is not None:
            min_z, max_z = min(min_z, max_z), max(min_z, max_z)

        return (min_x, max_x, min_y, max_y, min_z, max_z)

    def is_target_in_bed(self, x: float, y: float, z: float, bed_location=None) -> bool:
        """
        Checks if target coordinates (x, y, z) fall within the defined bed bounding box.
        """
        bounds = self.parse_bed_bounds(bed_location)
        if bounds is None:
            return False

        min_x, max_x, min_y, max_y, min_z, max_z = bounds

        # Check horizontal bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        # Check vertical bounds if specified
        if min_z is not None and max_z is not None:
            return min_z <= z <= max_z

        return True

    def classify_target(self, x: float, y: float, z: float, velocity: float, vy: float = 0.0, bed_location=None) -> dict:
        """
        Runs the decision tree on a single frame's tracked coordinates.
        
        :param x: Target X coordinate (meters)
        :param y: Target Y coordinate (meters)
        :param z: Target Z coordinate (meters - height from floor)
        :param velocity: Absolute velocity magnitude (m/s)
        :param vy: Velocity along the Y-axis (optional, used for door approach check)
        :param bed_location: Optional override for bed location bounding box
        :return: Dictionary containing the classified type, posture, approaching_door, and in_bed status
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
        active_bed = bed_location if bed_location is not None else self.bed_location
        in_bed = self.is_target_in_bed(x, y, z, active_bed)

        if velocity >= self.threshold_walking_min_v:
            # Target is actively moving
            posture = "WALKING"
        else:
            # Target is stationary (Velocity < threshold_walking_min_v)
            if active_bed is not None:
                if in_bed:
                    # Stationary inside bed area -> LAYING (unless standing tall on bed)
                    bounds = self.parse_bed_bounds(active_bed)
                    max_bed_z = bounds[5] if (bounds and bounds[5] is not None) else self.threshold_sitting_max_z
                    if z <= max_bed_z:
                        posture = "LAYING"
                    else:
                        posture = "STANDING"
                else:
                    # Stationary outside bed area -> SITTING or STANDING (not LAYING)
                    if z <= self.threshold_sitting_max_z:
                        posture = "SITTING"
                    else:
                        posture = "STANDING"
            else:
                # Default heuristic when no bed location is configured
                if z <= self.threshold_laying_max_z:
                    posture = "LAYING"
                elif self.threshold_sitting_min_z <= z <= self.threshold_sitting_max_z:
                    posture = "SITTING"
                else:
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
            "approaching_door": is_approaching_door,
            "in_bed": in_bed
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Define bed location: X from 0.0 to 1.5m, Y from 1.5 to 3.0m
    bed = [0.0, 1.5, 1.5, 3.0]
    classifier = KnightlyClassifier(bed_location=bed)
    
    # 1. Target stationary inside bed area -> LAYING
    result1 = classifier.classify_target(x=0.5, y=2.0, z=0.55, velocity=0.05)
    print(f"Test 1 (in bed): {result1}")
    
    # 2. Target stationary outside bed area -> SITTING
    result2 = classifier.classify_target(x=2.5, y=1.0, z=0.55, velocity=0.05)
    print(f"Test 2 (outside bed): {result2}")

    # 3. Target walking quickly towards the door
    result3 = classifier.classify_target(x=0.1, y=0.8, z=1.1, velocity=0.6, vy=-0.4)
    print(f"Test 3 (walking towards door): {result3}")
