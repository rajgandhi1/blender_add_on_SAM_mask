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

def update_use_texture(self, context):
    """Callback for texture toggle"""
    obj = bpy.data.objects.get(context.scene.projection_3d_object)
    if not obj:
        return
        
    # Create or get material
    mat_name = f"{obj.name}_material"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create basic nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    
    if self.use_texture and context.scene.projection_texture_map:
        # Load and apply texture
        tex_image = nodes.new('ShaderNodeTexImage')
        try:
            img = bpy.data.images.load(context.scene.projection_texture_map)
            tex_image.image = img
            links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        except Exception as e:
            print(f"Failed to load texture: {str(e)}")
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

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
    use_texture: BoolProperty(
        name="Use Texture",
        description="Use texture instead of solid color",
        default=False,
        update=update_use_texture
    )
    color: FloatVectorProperty(
        subtype='COLOR',
        default=(0.0, 1.0, 0.0),
        min=0.0,
        max=1.0
    )
    texture_path: StringProperty(
        name="Texture Path",
        subtype='FILE_PATH'
    )
    visible: BoolProperty(default=True)
    material: PointerProperty(type=bpy.types.Material)
    cameras: PointerProperty(type=MaterialIDCameraList)
    prompts: PointerProperty(type=MaterialIDPrompts)

    masks: PointerProperty(type=MaterialIDMasks)

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
            if item.use_texture:
                icon = 'TEXTURE'
            else:
                icon = 'COLOR'
            layout.prop(item, "visible", text="", icon=icon)

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
            
            row = box.row()
            row.prop(material, "use_texture", text="Use Texture")
            
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
            
            box.label(text="Generated Masks:")
            row = box.row()
            row.template_list("MASK_UL_List", "mask_list", 
                            material.masks, "masks",
                            material.masks, "mask_index")

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
    for idx, material in enumerate(context.scene.material_ids):
        material_empty_name = f"MaterialID_{material.name}"
        if material_empty_name in bpy.data.objects:
            material_empty = bpy.data.objects[material_empty_name]
            material_empty.hide_viewport = (idx != context.scene.material_id_index)
            for child in material_empty.children:
                child.hide_viewport = (idx != context.scene.material_id_index)
        
        # Update material visibility and texture
        obj = bpy.data.objects.get(context.scene.projection_3d_object)
        if obj and material.material:
            if idx == context.scene.material_id_index:
                # Update material nodes based on current material settings
                nodes = material.material.node_tree.nodes
                links = material.material.node_tree.links
                
                # Clear existing nodes
                nodes.clear()
                
                # Create basic nodes
                output = nodes.new('ShaderNodeOutputMaterial')
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                
                if material.use_texture and material.texture_path:
                    # Add texture node
                    tex_image = nodes.new('ShaderNodeTexImage')
                    try:
                        img = bpy.data.images.load(material.texture_path)
                        tex_image.image = img
                        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
                    except Exception as e:
                        print(f"Failed to load texture: {str(e)}")
                else:
                    # Use solid color
                    bsdf.inputs['Base Color'].default_value = material.color + (1.0,)
                
                links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
                
                # Ensure material is assigned to object
                if material.material.name not in obj.data.materials:
                    obj.data.materials.append(material.material)

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