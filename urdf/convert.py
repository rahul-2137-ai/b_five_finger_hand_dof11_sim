import os
import mujoco

def convert_urdf(urdf_filename, xml_filename):
    assets = {}
    mesh_root = "../meshes"
    
    # Read all STL files into memory as bytes for MuJoCo's assets
    for root, _, files in os.walk(mesh_root):
        for file in files:
            if file.lower().endswith(('.stl', '.obj')):
                full_path = os.path.join(root, file)
                
                # Reconstruct the exact 'package://...' string the URDF is looking for
                # e.g., 'package://revo2_description/meshes/revo2_left_hand/left_base_link.STL'
                relative_part = os.path.relpath(full_path, "..")
                asset_key = f"package://revo2_description/{relative_part}"
                
                with open(full_path, "rb") as f:
                    assets[asset_key] = f.read()

    try:
        # Load using the in-memory assets map
        model = mujoco.MjModel.from_xml_path(urdf_filename, assets=assets)
        mujoco.mj_saveLastXML(xml_filename, model)
        print(f"Successfully converted {urdf_filename} -> {xml_filename}")
    except Exception as e:
        print(f"Failed to convert {urdf_filename}: {e}")

# Convert both hands
convert_urdf("revo2_left_hand.urdf", "revo2_left_hand.xml")
convert_urdf("revo2_right_hand.urdf", "revo2_right_hand.xml")
