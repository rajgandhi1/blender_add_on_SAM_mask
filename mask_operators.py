import bpy
import os
from bpy.types import Operator
from bpy.props import BoolProperty, StringProperty
import numpy as np

class OBJECT_OT_BakeCombinedMask(Operator):
    bl_idname = "wm.bake_combined_mask"
    bl_label = "Bake Combined Mask"
    
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
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def bake_material_masks(self, context, material_id, combined_image):
        """Bake all masks for a specific material ID into a UV texture and combine them"""
        # Get the target mesh object
        obj = bpy.data.objects.get(context.scene.projection_3d_object)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No valid mesh object found")
            return None

        # Ensure the object is in Object Mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Ensure the object has an active UV map
        if not obj.data.uv_layers:
            self.report({'ERROR'}, f"Object '{obj.name}' does not have any UV maps. Please create one.")
            return None
        obj.data.uv_layers.active_index = 0

        # Assign the image to all materials in the object
        for mat_slot in obj.material_slots:
            if mat_slot.material:
                mat = mat_slot.material
                mat.use_nodes = True
                nodes = mat.node_tree.nodes

                # Create or reuse an image texture node
                img_node = nodes.get("BakedTexture")
                if not img_node:
                    img_node = nodes.new(type='ShaderNodeTexImage')
                    img_node.name = "BakedTexture"

                img_node.image = combined_image
                mat.node_tree.nodes.active = img_node

        # Set baking settings
        bpy.context.scene.cycles.bake_type = 'COMBINED'
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128

        # Select the object in the correct context
        for obj_iter in bpy.context.view_layer.objects:
            obj_iter.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Perform baking
        bpy.ops.object.bake(type='COMBINED')

        # Blend the baked image into the combined image
        combined_pixels = np.array(combined_image.pixels[:])
        baked_pixels = np.array(combined_image.pixels[:])

        # Blend logic: prefer color over black
        combined_pixels = np.maximum(combined_pixels, baked_pixels)

        # Update combined image with blended pixels
        combined_image.pixels = combined_pixels

    def execute(self, context):
        obj = bpy.data.objects.get(context.scene.projection_3d_object)
        if not obj:
            self.report({'ERROR'}, "No target object selected")
            return {'CANCELLED'}

        # Ensure filepath has .png extension
        if not self.filepath.lower().endswith('.png'):
            self.filepath = self.filepath + '.png'

        # Create a new image for the combined result
        res_x = context.scene.projection_resolution_x
        res_y = context.scene.projection_resolution_y
        combined_image = bpy.data.images.new(
            name="combined_baked_texture",
            width=res_x,
            height=res_y,
            alpha=True
        )

        # First pass: Bake each material ID's masks and combine them
        for material_id in context.scene.material_ids:
            if not material_id.visible:
                continue

            self.bake_material_masks(context, material_id, combined_image)

        # Save the combined image with proper extension
        combined_image.filepath_raw = self.filepath
        combined_image.file_format = 'PNG'
        combined_image.save()

        self.report({'INFO'}, f"Combined baking complete. Image saved as '{self.filepath}'.")

        # Clean up
        bpy.data.images.remove(combined_image)

        return {'FINISHED'}

class OBJECT_OT_ToggleCombinedMask(Operator):
    bl_idname = "wm.toggle_combined_mask"
    bl_label = "Toggle Combined Mask"
    
    def execute(self, context):
        obj = bpy.data.objects.get(context.scene.projection_3d_object)
        if not obj:
            self.report({'ERROR'}, "No target object selected")
            return {'CANCELLED'}
            
        # Toggle visibility of all material masks
        for material_id in context.scene.material_ids:
            if material_id.baked_texture:
                material_id.baked_texture.use_fake_user = True
                # Toggle visibility logic here
                # You might want to implement specific visibility toggling based on your needs
                
        self.report({'INFO'}, "Toggled combined mask visibility")
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