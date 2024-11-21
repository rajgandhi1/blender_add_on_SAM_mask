import bpy
import os
from bpy.types import Operator
import numpy as np
import gpu
from gpu_extras.batch import batch_for_shader
from bpy.props import BoolProperty, StringProperty

class OBJECT_OT_BakeCombinedMask(Operator):
    bl_idname = "wm.bake_combined_mask"
    bl_label = "Bake Combined Mask"
    
    # Add file selector properties
    filepath: StringProperty(
        name="Save Path",
        description="Path to save the combined mask",
        default="",
        maxlen=1024,
        subtype='FILE_PATH'
    )
    
    filter_glob: StringProperty(
        default='*.png',
        options={'HIDDEN'},
        maxlen=255,
    )

    def invoke(self, context, event):
        # Open file selector when button is pressed
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def create_mask_material(self, mask_image):
        mat_name = "CombinedMaskMaterial"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        tex_image = nodes.new('ShaderNodeTexImage')
        
        tex_image.image = mask_image
        
        links.new(tex_image.outputs[0], bsdf.inputs[0])
        links.new(bsdf.outputs[0], output.inputs[0])
        
        return mat

    def combine_masks(self, context):
        scene = context.scene
        obj = bpy.data.objects.get(scene.projection_3d_object)
        
        if not obj:
            self.report({'ERROR'}, "No 3D object selected")
            return None
            
        res_x = scene.projection_resolution_x
        res_y = scene.projection_resolution_y
        pixel_count = res_x * res_y * 4  # 4 channels (RGBA)
        
        combined_image = bpy.data.images.new(
            name="CombinedMask",
            width=res_x,
            height=res_y
        )
        
        # Initialize pixel array with the correct shape
        pixels = np.zeros(pixel_count, dtype=np.float32)
        pixels = pixels.reshape(res_y, res_x, 4)  # Reshape to 2D array with RGBA channels
        
        for idx, material_id in enumerate(scene.material_ids):
            if not material_id.visible:
                continue
                
            if material_id.use_texture and material_id.texture_path:
                if os.path.exists(bpy.path.abspath(material_id.texture_path)):
                    img = bpy.data.images.load(bpy.path.abspath(material_id.texture_path))
                    tex_pixels = np.array(img.pixels[:])
                    tex_pixels = tex_pixels.reshape(img.size[1], img.size[0], 4)
                    
                    # Ensure texture is the right size
                    if img.size[0] != res_x or img.size[1] != res_y:
                        from skimage.transform import resize
                        tex_pixels = resize(tex_pixels, (res_y, res_x, 4), preserve_range=True)
                    
                    # Add to combined mask
                    pixels += tex_pixels * (idx + 1)
            else:
                # Create color array with correct shape
                color = np.array(material_id.color)
                color_rgba = np.append(color, 1.0)  # Add alpha channel
                color_array = np.tile(color_rgba, (res_y, res_x, 1))
                pixels += color_array * (idx + 1)
        
        # Normalize pixels
        if len(scene.material_ids) > 0:
            max_val = np.max(pixels)
            if max_val > 0:
                pixels = pixels / max_val
        
        # Flatten and set image pixels
        pixels = pixels.flatten()
        combined_image.pixels.foreach_set(pixels)
        
        return combined_image

    def execute(self, context):
        combined_image = self.combine_masks(context)
        if not combined_image:
            return {'CANCELLED'}
        
        mask_material = self.create_mask_material(combined_image)
        
        obj = bpy.data.objects.get(context.scene.projection_3d_object)
        if not obj:
            self.report({'ERROR'}, "No 3D object selected")
            return {'CANCELLED'}
        
        if mask_material.name not in obj.data.materials:
            obj.data.materials.append(mask_material)
        
        # Save the combined mask to user-specified path
        try:
            # Ensure the filepath ends with .png
            filepath = self.filepath
            if not filepath.lower().endswith('.png'):
                filepath += '.png'
            
            combined_image.save_render(filepath)
            self.report({'INFO'}, f"Combined mask saved successfully to {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Error saving combined mask: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

class OBJECT_OT_ToggleCombinedMask(Operator):
    bl_idname = "wm.toggle_combined_mask"
    bl_label = "Toggle Combined Mask"
    
    # Store visibility state as a scene property instead of class variable
    def store_original_materials(self, obj):
        # Store as a string property on the object
        if not obj.get('original_materials'):
            original_mats = [mat.name if mat else "" for mat in obj.data.materials]
            obj['original_materials'] = ','.join(original_mats)
            obj['showing_mask'] = False
    
    def execute(self, context):
        obj = bpy.data.objects.get(context.scene.projection_3d_object)
        if not obj:
            self.report({'ERROR'}, "No 3D object selected")
            return {'CANCELLED'}
        
        # Find or create the mask material
        mask_material = bpy.data.materials.get("CombinedMaskMaterial")
        if not mask_material:
            self.report({'WARNING'}, "No combined mask material found. Please bake mask first.")
            return {'CANCELLED'}
        
        # Initialize material storage if needed
        self.store_original_materials(obj)
        
        # Get current state
        showing_mask = obj.get('showing_mask', False)
        
        if not showing_mask:
            # Store and clear current materials
            if not obj.get('original_materials'):
                original_mats = [mat.name if mat else "" for mat in obj.data.materials]
                obj['original_materials'] = ','.join(original_mats)
            
            # Clear existing materials
            obj.data.materials.clear()
            
            # Apply mask material
            obj.data.materials.append(mask_material)
            obj['showing_mask'] = True
            
        else:
            # Restore original materials
            obj.data.materials.clear()
            
            # Get original materials from stored string
            original_mat_names = obj.get('original_materials', '').split(',')
            for mat_name in original_mat_names:
                if mat_name and mat_name in bpy.data.materials:
                    obj.data.materials.append(bpy.data.materials[mat_name])
            
            obj['showing_mask'] = False
        
        # Force viewport update
        obj.data.update()
        context.view_layer.update()
        
        self.report({'INFO'}, 
                   "Showing combined mask" if obj['showing_mask'] 
                   else "Showing original materials")
        
        return {'FINISHED'}

# Registration
classes = (
    OBJECT_OT_BakeCombinedMask,
    OBJECT_OT_ToggleCombinedMask,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()