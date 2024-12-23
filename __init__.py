bl_info = {
    "name": "Projection Tool",
    "blender": (2, 80, 0),
    "category": "3D View",
    "author": "Raj Gandhi",
    "version": (2, 0),
    "location": "View3D > Sidebar > Projection Tool",
    "description": "Projection tool for creating texture using masking (SAM)",
}

import bpy
from bpy.props import (StringProperty, EnumProperty, IntProperty, 
                      FloatVectorProperty, BoolProperty, CollectionProperty, 
                      PointerProperty)
from bpy.types import PropertyGroup, UIList, Operator
from . import operators
from . import mask_operators

# Property Groups
class CameraProperties(PropertyGroup):
    name: StringProperty(default="Camera")
    active: BoolProperty(default=True)

class PromptProperties(PropertyGroup):
    name: StringProperty(default="")
    coordinate: FloatVectorProperty(size=2)
    face_index: IntProperty(default=-1)

class MaterialIDCameraList(PropertyGroup):
    cameras: CollectionProperty(type=CameraProperties)
    camera_index: IntProperty()

class MaterialIDPrompts(PropertyGroup):
    positive_prompts: CollectionProperty(type=PromptProperties)
    positive_prompt_index: IntProperty()
    negative_prompts: CollectionProperty(type=PromptProperties)
    negative_prompt_index: IntProperty()

def update_material_name(self, context):
    """Callback for material name changes"""
    # Update collection name if it exists
    old_coll_name = f"MaterialID_{self.get('previous_name', self.name)}_Collection"
    new_coll_name = f"MaterialID_{self.name}_Collection"
    
    if old_coll_name in bpy.data.collections:
        collection = bpy.data.collections[old_coll_name]
        collection.name = new_coll_name
        
    # Update empty name if it exists
    old_empty_name = f"MaterialID_{self.get('previous_name', self.name)}"
    new_empty_name = f"MaterialID_{self.name}"
    
    if old_empty_name in bpy.data.objects:
        empty = bpy.data.objects[old_empty_name]
        empty.name = new_empty_name
    
    # Update material name if it exists
    if self.material:
        self.material.name = self.name
    
    # Store the current name for future reference
    self['previous_name'] = self.name

class MaskProperties(PropertyGroup):
    name: StringProperty(default="Mask")
    material: PointerProperty(type=bpy.types.Material)
    image: PointerProperty(type=bpy.types.Image)

class MaterialIDMasks(PropertyGroup):
    masks: CollectionProperty(type=MaskProperties)
    mask_index: IntProperty()

class MaterialIDProperties(PropertyGroup):
    name: StringProperty(
        default="Material",
        update=update_material_name
    )
    color: FloatVectorProperty(
        subtype='COLOR',
        default=(0.0, 1.0, 0.0),
        min=0.0,
        max=1.0
    )
    texture_path: StringProperty(
        name="Texture Path",
        description="Path to texture file",
        default="",
        maxlen=1024,
        subtype='FILE_PATH'
    )
    visible: BoolProperty(default=True)
    material: PointerProperty(type=bpy.types.Material)
    cameras: PointerProperty(type=MaterialIDCameraList)
    prompts: PointerProperty(type=MaterialIDPrompts)
    masks: PointerProperty(type=MaterialIDMasks)
    baked_texture: PointerProperty(type=bpy.types.Image)

class MASK_UL_List(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False)

# List Templates
class CAMERA_UL_List(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False)
            layout.prop(item, "active", text="")

class MATERIAL_UL_List(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False)
            layout.prop(item, "visible", text="", icon='COLOR')

class PROMPT_UL_List(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row()
            row.prop(item, "name", text="", emboss=False)
            if item.face_index >= 0:
                row.label(text=f"Face: {item.face_index}")
            # Add delete button with proper error checking
            delete_op = row.operator("wm.delete_point", text="", icon='X')
            if delete_op:  # Check if operator exists
                delete_op.point_name = item.name

# Main Panel
class ProjectionToolPanel(bpy.types.Panel):
    bl_label = "Projection Tool UI"
    bl_idname = "VIEW3D_PT_projection_tool_ui"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Projection Tool"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Scene Data Section
        box = layout.box()
        box.label(text="Scene Data")
        
        row = box.row()
        row.prop(scene, "projection_3d_object", text="3D Object")
        row.operator("wm.select_3d_object", text="", icon='FILE_FOLDER')

        # Modified texture map selection to use a single row with split
        row = box.row()
        split = row.split(factor=1)  # Adjust factor to control button size
        split.prop(scene, "projection_texture_map", text="Texture Map")
        # split.operator("wm.select_texture_map", text="", icon='FILE_FOLDER')
        
        box.prop(scene, "projection_material", text="Material")
        
        row = box.row()
        row.prop(scene, "projection_resolution_x", text="Resolution")
        row.prop(scene, "projection_resolution_y", text="x")

        # Material IDs List Section
        box = layout.box()
        box.label(text="Material IDs")
        
        row = box.row()
        row.template_list("MATERIAL_UL_List", "material_ids", scene, 
                         "material_ids", scene, "material_id_index")
        
        col = row.column(align=True)
        col.operator("wm.add_material_id", text="", icon='ADD')
        col.operator("wm.delete_material_id", text="", icon='REMOVE')
        
        row = box.row()
        row.operator("wm.bake_combined_mask", text="Bake Combined Mask")
        row.operator("wm.toggle_combined_mask", text="Toggle Combined Mask")

        # Selected Material ID Settings
        if len(scene.material_ids) > 0 and scene.material_id_index >= 0:
            material = scene.material_ids[scene.material_id_index]
            print(f"Drawing settings for material: {material.name}")  # Debug print
            
            box = layout.box()
            box.label(text=f"Material ID: {material.name}")
            
            # Material appearance settings
            row = box.row()
            row.prop(material, "name", text="Name")
            row.prop(material, "visible", text="Visible")
            
            box.prop(material, "color", text="Color")
            
            # Render Mask button moved above Add Camera
            box.operator("wm.render_mask", text="Render Mask")
            
            # Camera controls
            row = box.row()
            row.operator("wm.add_camera", text="Add Camera").material_index = scene.material_id_index
            
            box.label(text="Cameras:")
            row = box.row()
            row.template_list("CAMERA_UL_List", "camera_list", 
                            material.cameras, "cameras",
                            material.cameras, "camera_index")
            
            col = row.column(align=True)
            col.operator("wm.add_camera", text="", icon='ADD').material_index = scene.material_id_index
            col.operator("wm.delete_camera", text="", icon='REMOVE').material_index = scene.material_id_index
            
            # Prompt controls
            row = box.row()
            row.operator("wm.add_positive_point", text="Add +Point").material_index = scene.material_id_index
            row.operator("wm.add_negative_point", text="Add -Point").material_index = scene.material_id_index
            
            box.label(text="Positive Prompts:")
            row = box.row()
            row.template_list("PROMPT_UL_List", "positive_prompts",
                            material.prompts, "positive_prompts",
                            material.prompts, "positive_prompt_index")
            
            box.label(text="Negative Prompts:")
            row = box.row()
            row.template_list("PROMPT_UL_List", "negative_prompts",
                            material.prompts, "negative_prompts",
                            material.prompts, "negative_prompt_index")

# Registration
classes = (
    PromptProperties,
    CameraProperties,
    MaterialIDCameraList,
    MaterialIDPrompts,
    MaskProperties,
    MaterialIDMasks,
    MaterialIDProperties,
    CAMERA_UL_List,
    MATERIAL_UL_List,
    PROMPT_UL_List,
    MASK_UL_List,
    ProjectionToolPanel,
)

def update_material_selection(self, context):
    """Callback for material selection changes"""
    active_index = context.scene.material_id_index
    
    # First, hide ALL objects in ALL material collections
    for material in context.scene.material_ids:
        collection_name = f"MaterialID_{material.name}_Collection"
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
            
            # Hide the collection itself
            collection.hide_viewport = True
            collection.hide_render = True
            
            # Hide all objects in the collection
            for obj in collection.objects:
                if obj.type in {'CAMERA', 'MESH'}:  # Cameras and iconospheres (prompt points)
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.hide_select = True
                    if hasattr(obj, "hide_set"):
                        obj.hide_set(True)
    
    # Then, show ONLY the active material's objects
    if active_index >= 0 and active_index < len(context.scene.material_ids):
        active_material = context.scene.material_ids[active_index]
        active_collection_name = f"MaterialID_{active_material.name}_Collection"
        
        if active_collection_name in bpy.data.collections:
            active_collection = bpy.data.collections[active_collection_name]
            
            # Show the active collection
            active_collection.hide_viewport = False
            active_collection.hide_render = False
            
            # Show all objects in the active collection
            for obj in active_collection.objects:
                if obj.type in {'CAMERA', 'MESH'}:  # Cameras and iconospheres (prompt points)
                    obj.hide_viewport = False
                    obj.hide_render = False
                    obj.hide_select = False
                    if hasattr(obj, "hide_set"):
                        obj.hide_set(False)
    
    # Force viewport update
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def register():
    mask_operators.register()
    operators.register()
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register properties
    bpy.types.Scene.projection_3d_object = StringProperty(name="3D Object")
    bpy.types.Scene.projection_material = EnumProperty(
        name="Material",
        items=[
            ("TEXTURE", "Texture", ""),
            ("MATERIAL", "Material", "")
        ]
    )
    bpy.types.Scene.projection_texture_map = StringProperty(
        name="Texture Map",
        description="Texture map for the 3D object",
        subtype='FILE_PATH'
    )
    bpy.types.Scene.projection_resolution_x = IntProperty(name="X", default=512)
    bpy.types.Scene.projection_resolution_y = IntProperty(name="Y", default=512)

    bpy.types.Scene.material_ids = CollectionProperty(type=MaterialIDProperties)
    bpy.types.Scene.material_id_index = IntProperty(
        default=0,  # Set default to 0
        min=0,  # Ensure it can't go below 0
        update=update_material_selection
    )

    bpy.types.Scene.bake_resolution = IntProperty(
        name="Bake Resolution",
        default=1024,
        min=64,
        max=8192
    )

def unregister():
    operators.unregister()
    mask_operators.unregister()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.projection_3d_object
    del bpy.types.Scene.projection_material
    del bpy.types.Scene.projection_texture_map
    del bpy.types.Scene.projection_resolution_x
    del bpy.types.Scene.projection_resolution_y
    del bpy.types.Scene.projection_server_url
    del bpy.types.Scene.material_ids
    del bpy.types.Scene.material_id_index

if __name__ == "__main__":
    register()