import bpy
from bpy.types import Operator
import math
import mathutils
import bmesh
from bpy.props import BoolProperty, StringProperty, IntProperty  # Ensure StringProperty is importedimport gpu
import bgl
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
import os
from . import mask_operators
from bpy_extras.object_utils import world_to_camera_view

class OBJECT_OT_Select3DObject(Operator):
    bl_idname = "wm.select_3d_object"
    bl_label = "Select 3D Object"
    
    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'MESH':
            context.scene.projection_3d_object = obj.name
            self.report({'INFO'}, f"Selected object: {obj.name}")
        else:
            self.report({'WARNING'}, "Please select a valid mesh object")
        return {'FINISHED'}

class OBJECT_OT_AddMaterialID(Operator):
    bl_idname = "wm.add_material_id"
    bl_label = "Add Material ID"
    
    def execute(self, context):
        material_id = context.scene.material_ids.add()
        material_id.name = f"Material {len(context.scene.material_ids)}"
        
        # Create new material
        mat = bpy.data.materials.new(name=material_id.name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear default nodes
        nodes.clear()
        
        # Create basic material nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        
        # Link nodes
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        # Store material reference
        material_id.material = mat
        
        # Set the new material as active
        context.scene.material_id_index = len(context.scene.material_ids) - 1
        
        return {'FINISHED'}

class OBJECT_OT_DeleteMaterialID(Operator):
    bl_idname = "wm.delete_material_id"
    bl_label = "Delete Material ID"
    
    def execute(self, context):
        idx = context.scene.material_id_index
        if idx >= 0 and idx < len(context.scene.material_ids):
            material = context.scene.material_ids[idx]
            
            # Delete material collection and all its contents
            material_coll_name = f"MaterialID_{material.name}_Collection"
            if material_coll_name in bpy.data.collections:
                collection = bpy.data.collections[material_coll_name]
                # Remove all objects in the collection
                for obj in collection.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
                # Remove the collection itself
                bpy.data.collections.remove(collection)
            
            # Remove material from Blender
            if material.material:
                bpy.data.materials.remove(material.material)
            
            context.scene.material_ids.remove(idx)
            context.scene.material_id_index = min(max(0, idx), len(context.scene.material_ids) - 1)
        return {'FINISHED'}

class OBJECT_OT_AddCamera(Operator):
    bl_idname = "wm.add_camera"
    bl_label = "Add Camera"
    
    material_index: IntProperty()
    
    def execute(self, context):
        material = context.scene.material_ids[self.material_index]
        
        # Create new camera
        camera_data = bpy.data.cameras.new(name="Camera")
        camera = bpy.data.objects.new(name="Camera", object_data=camera_data)
        
        # Get the current viewport's view matrix and perspective settings
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                view = area.spaces.active.region_3d
                if view.view_perspective == 'CAMERA':
                    # If in camera view, use the active camera's matrix
                    view_camera = context.scene.camera
                    if view_camera:
                        camera.matrix_world = view_camera.matrix_world.copy()
                else:
                    # Convert view matrix to world space matrix
                    view_matrix = view.view_matrix.inverted()
                    camera.matrix_world = view_matrix
                    
                    # Set default camera settings
                    camera.data.lens = 50  # Standard 50mm focal length
                    # Use scene clipping values instead of view properties
                    camera.data.clip_start = 0.1  # Default near clip
                    camera.data.clip_end = 1000.0  # Default far clip
                    
                    # If in orthographic view, match those settings
                    if view.is_perspective:
                        camera.data.type = 'PERSP'
                    else:
                        camera.data.type = 'ORTHO'
                        camera.data.ortho_scale = view.view_distance * 2
                break
        
        # Add camera to material's list
        cam_item = material.cameras.cameras.add()
        cam_item.name = camera.name
        cam_item.active = True
        
        # Create or get material ID empty and its collection
        material_empty_name = f"MaterialID_{material.name}"
        material_coll_name = f"MaterialID_{material.name}_Collection"
        
        if material_coll_name not in bpy.data.collections:
            material_collection = bpy.data.collections.new(material_coll_name)
            context.scene.collection.children.link(material_collection)
        else:
            material_collection = bpy.data.collections[material_coll_name]
        
        if material_empty_name not in bpy.data.objects:
            material_empty = bpy.data.objects.new(material_empty_name, None)
            material_empty.empty_display_type = 'CUBE'
            material_empty.empty_display_size = 0.1
            material_collection.objects.link(material_empty)
        else:
            material_empty = bpy.data.objects[material_empty_name]
        
        # Add camera to the material collection only
        material_collection.objects.link(camera)
        
        # Parent camera to material ID empty
        camera.parent = material_empty
        
        # Update visibility based on material selection
        material_empty.hide_viewport = (context.scene.material_id_index != self.material_index)
        camera.hide_viewport = material_empty.hide_viewport
        
        self.report({'INFO'}, f"Added camera: {camera.name}")
        return {'FINISHED'}
    
class OBJECT_OT_DeleteCamera(Operator):
    bl_idname = "wm.delete_camera"
    bl_label = "Delete Camera"
    
    material_index: IntProperty()
    
    def execute(self, context):
        material = context.scene.material_ids[self.material_index]
        idx = material.cameras.camera_index
        
        if idx >= 0 and idx < len(material.cameras.cameras):
            camera_name = material.cameras.cameras[idx].name
            if camera_name in bpy.data.objects:
                camera = bpy.data.objects[camera_name]
                bpy.data.objects.remove(camera, do_unlink=True)
            
            material.cameras.cameras.remove(idx)
            material.cameras.camera_index = min(max(0, idx), len(material.cameras.cameras) - 1)
            
            self.report({'INFO'}, f"Deleted camera: {camera_name}")
        return {'FINISHED'}

class OBJECT_OT_AddPromptBase(Operator):
    bl_idname = "wm.add_prompt_base"
    bl_label = "Add Prompt Base"
    
    material_index: IntProperty()
    is_positive: BoolProperty()
    
    def create_icosphere(self, context, location, color, prompt_name):
        # Create mesh and object
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.1)
        mesh = bpy.data.meshes.new(prompt_name)
        bm.to_mesh(mesh)
        bm.free()
        
        ico = bpy.data.objects.new(prompt_name, mesh)
        ico.location = location
        
        # Create material for the icosphere
        mat = bpy.data.materials.new(name=f"{prompt_name}_Material")
        mat.use_nodes = False
        mat.diffuse_color = color + (1,)  # Add alpha value
        ico.data.materials.append(mat)
        
        # Get or create the material collection
        material = context.scene.material_ids[self.material_index]
        material_coll_name = f"MaterialID_{material.name}_Collection"
        
        if material_coll_name not in bpy.data.collections:
            material_collection = bpy.data.collections.new(material_coll_name)
            context.scene.collection.children.link(material_collection)
        else:
            material_collection = bpy.data.collections[material_coll_name]
        
        # Add icosphere to the material collection only
        material_collection.objects.link(ico)
        
        return ico
    
    def modal(self, context, event):
        context.area.header_text_set("Click on a face to add a {} point. Right click or ESC to cancel.".format(
            "positive" if self.is_positive else "negative"
        ))
        
        if event.type == 'MOUSEMOVE':
            self.mouse_pos = (event.mouse_region_x, event.mouse_region_y)
            context.area.tag_redraw()
            
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            obj = bpy.data.objects.get(context.scene.projection_3d_object)
            if not obj:
                self.report({'WARNING'}, "No valid object selected")
                context.area.header_text_set(None)
                return {'CANCELLED'}
            
            # Get face under mouse
            mouse_coord = event.mouse_region_x, event.mouse_region_y
            region = context.region
            rv3d = context.region_data
            
            view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_coord)
            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_coord)
            
            matrix = obj.matrix_world
            matrix_inv = matrix.inverted()
            ray_origin_obj = matrix_inv @ ray_origin
            ray_dir_obj = matrix_inv.to_3x3() @ view_vector
            
            success, location, normal, face_idx = obj.ray_cast(ray_origin_obj, ray_dir_obj)
            
            if success:
                material = context.scene.material_ids[self.material_index]
                prompts = material.prompts.positive_prompts if self.is_positive else material.prompts.negative_prompts
                
                # Create prompt and set name
                prompt = prompts.add()
                prompt_name = f"{'+'if self.is_positive else '-'}Point{len(prompts)}"
                prompt.name = prompt_name
                prompt.face_index = face_idx
                
                # Create visual marker
                world_loc = matrix @ location
                color = (0.0, 1.0, 0.0) if self.is_positive else (1.0, 0.0, 0.0)
                ico = self.create_icosphere(context, world_loc, color, prompt_name)
                
                # Get or create material ID empty
                material_empty_name = f"MaterialID_{material.name}"
                if material_empty_name not in bpy.data.objects:
                    material_empty = bpy.data.objects.new(material_empty_name, None)
                    material_empty.empty_display_type = 'CUBE'
                    material_empty.empty_display_size = 0.1
                    context.scene.collection.objects.link(material_empty)
                else:
                    material_empty = bpy.data.objects[material_empty_name]
                
                # Parent icosphere to material ID empty
                ico.parent = material_empty
                
                # Update visibility based on material selection
                material_empty.hide_viewport = (context.scene.material_id_index != self.material_index)
                
                # Apply material or texture to the face
                self.apply_material_to_face(context, obj, face_idx, material)
                
                context.area.header_text_set(None)
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                return {'FINISHED'}
            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            # Add the drawing handler
            args = (self, context)
            self._handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback_px, args, 'WINDOW', 'POST_PIXEL'
            )
            
            self.mouse_pos = (event.mouse_region_x, event.mouse_region_y)
            
            # Set initial instruction
            context.area.header_text_set("Click on a face to add a {} point. Right click or ESC to cancel.".format(
                "positive" if self.is_positive else "negative"
            ))
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}
    
    def draw_callback_px(self, context):
        # Draw cursor
        shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        bgl.glEnable(bgl.GL_BLEND)
        
        # Draw circle
        vertices = [(self.mouse_pos[0] + math.cos(angle) * 10, 
                    self.mouse_pos[1] + math.sin(angle) * 10) 
                   for angle in range(0, 360, 30)]
        
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
        
        shader.bind()
        color = (0.0, 1.0, 0.0, 1.0) if self.is_positive else (1.0, 0.0, 0.0, 1.0)
        shader.uniform_float("color", color)
        batch.draw(shader)
        
        bgl.glDisable(bgl.GL_BLEND)
    
    def apply_material_to_face(self, context, obj, face_idx, material_id):
        if not obj.data.materials:
            # Create a default material if none exists
            default_mat = bpy.data.materials.new(name="Default")
            obj.data.materials.append(default_mat)
        
        # Ensure the material is in the object's material slots
        if material_id.material.name not in obj.data.materials:
            obj.data.materials.append(material_id.material)
        
        material_index = obj.data.materials.find(material_id.material.name)
        
        # Update material nodes based on material ID settings
        self.update_material_nodes(material_id)
        
        # Assign material to face
        obj.data.polygons[face_idx].material_index = material_index
    
    def update_material_nodes(self, material_id):
        mat = material_id.material
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes
        nodes.clear()
        
        # Create output and shader nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        
        # Position nodes
        output.location = (300, 0)
        principled.location = (0, 0)
        
        # Link shader to output
        links.new(principled.outputs[0], output.inputs[0])
        
        if material_id.use_texture and material_id.texture_path:
            # Create and setup texture node
            tex_image = nodes.new('ShaderNodeTexImage')
            tex_image.location = (-300, 0)
            
            # Load texture image
            if os.path.exists(bpy.path.abspath(material_id.texture_path)):
                try:
                    img = bpy.data.images.load(bpy.path.abspath(material_id.texture_path))
                    tex_image.image = img
                except:
                    self.report({'WARNING'}, f"Could not load texture: {material_id.texture_path}")
            
            # Link texture to shader
            links.new(tex_image.outputs[0], principled.inputs[0])  # Base Color
        else:
            # Convert color to tuple and add alpha
            color = tuple(material_id.color) + (1.0,)
            principled.inputs[0].default_value = color

class OBJECT_OT_AddPositivePoint(OBJECT_OT_AddPromptBase):
    bl_idname = "wm.add_positive_point"
    bl_label = "Add Positive Point"
    
    def __init__(self):
        super().__init__()
        self.is_positive = True

class OBJECT_OT_AddNegativePoint(OBJECT_OT_AddPromptBase):
    bl_idname = "wm.add_negative_point"
    bl_label = "Add Negative Point"
    
    def __init__(self):
        super().__init__()
        self.is_positive = False

class OBJECT_OT_RenderMask(Operator):
    bl_idname = "wm.render_mask"
    bl_label = "Render Mask"

    filepath: StringProperty(
        name="Save Path",
        description="Path to save rendered images",
        default="",
        maxlen=1024,
        subtype='DIR_PATH'
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        try:
            # Base directory (user selected)
            base_dir = os.path.abspath(self.filepath)
            if not os.path.exists(base_dir):
                self.report({'ERROR'}, f"Base directory does not exist: {base_dir}")
                return {'CANCELLED'}

            # Create main subdirectories at the root level
            sam_model_dir = os.path.join(base_dir, "sam_model")
            os.makedirs(sam_model_dir, exist_ok=True)

            # Process all materials
            for material_index, material in enumerate(context.scene.material_ids):
                # Create material-specific directory
                material_dir = os.path.join(base_dir, f"material_{material.name}")
                os.makedirs(material_dir, exist_ok=True)

                # Create masks and renders subdirectories within material directory
                material_masks_dir = os.path.join(material_dir, "masks")
                material_renders_dir = os.path.join(material_dir, "renders")
                os.makedirs(material_masks_dir, exist_ok=True)
                os.makedirs(material_renders_dir, exist_ok=True)

                # Get iconosphere points for SAM
                pos_points = []
                neg_points = []
                
                # Get positive points (green iconospheres)
                for point in material.prompts.positive_prompts:
                    if point.name in bpy.data.objects:
                        obj = bpy.data.objects[point.name]
                        pos_points.append(obj.location)
                        self.report({'INFO'}, f"Found positive prompt at {obj.location}")
                        
                # Get negative points (red iconospheres)
                for point in material.prompts.negative_prompts:
                    if point.name in bpy.data.objects:
                        obj = bpy.data.objects[point.name]
                        neg_points.append(obj.location)
                        self.report({'INFO'}, f"Found negative prompt at {obj.location}")

                if not pos_points and not neg_points:
                    self.report({'WARNING'}, f"No prompt points found for material {material.name}, skipping...")
                    continue

                # Process each camera for this material
                for cam_index, cam in enumerate(material.cameras.cameras):
                    if not cam.active or cam.name not in bpy.data.objects:
                        continue

                    self.report({'INFO'}, f"Processing camera {cam_index + 1}/{len(material.cameras.cameras)}: {cam.name}")

                    # Set active camera
                    camera_obj = bpy.data.objects[cam.name]
                    context.scene.camera = camera_obj

                    # Define paths for this camera
                    render_name = f"{cam.name}.png"
                    render_path = os.path.join(material_renders_dir, render_name)
                    mask_path = os.path.join(material_masks_dir, f"{cam.name}_mask.png")

                    # Render image
                    self.report({'INFO'}, f"Rendering to: {render_path}")
                    bpy.context.scene.render.filepath = render_path
                    bpy.ops.render.render(write_still=True)

                    if not os.path.exists(render_path):
                        self.report({'ERROR'}, f"Render failed: {render_path}")
                        continue

                    # Convert 3D points to 2D coordinates for SAM
                    def convert_world_to_camera_view(scene, camera, point):
                        co_2d = bpy.context.scene.view_layers[0].depsgraph.scene.camera.matrix_world.normalized()
                        co_2d = world_to_camera_view(scene, camera, point)
                        return co_2d

                    # Process points for SAM
                    sam_points = []
                    sam_labels = []

                    # Process positive points
                    for point in pos_points:
                        coord_2d = convert_world_to_camera_view(context.scene, camera_obj, point)
                        if 0 <= coord_2d[0] <= 1 and 0 <= coord_2d[1] <= 1:  # Check if point is in view
                            sam_points.append(coord_2d)
                            sam_labels.append(1)  # Positive label
                            self.report({'INFO'}, f"Added positive SAM point at {coord_2d[0]:.3f}, {coord_2d[1]:.3f}")

                    # Process negative points
                    for point in neg_points:
                        coord_2d = convert_world_to_camera_view(context.scene, camera_obj, point)
                        if 0 <= coord_2d[0] <= 1 and 0 <= coord_2d[1] <= 1:  # Check if point is in view
                            sam_points.append(coord_2d)
                            sam_labels.append(0)  # Negative label
                            self.report({'INFO'}, f"Added negative SAM point at {coord_2d[0]:.3f}, {coord_2d[1]:.3f}")

                    if not sam_points:
                        self.report({'WARNING'}, f"No points visible in camera {cam.name}, skipping...")
                        continue

                    # Process with SAM
                    if self.process_with_sam_pipeline(sam_model_dir, render_path, sam_points, sam_labels, mask_path):
                        self.report({'INFO'}, f"Successfully processed {cam.name} for material {material.name}")
                    else:
                        self.report({'ERROR'}, f"Failed to process {cam.name} for material {material.name}")

            self.report({'INFO'}, "Completed processing all materials")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}

    def process_with_sam_pipeline(self, sam_model_dir, render_path, input_points, input_labels, mask_path):
        try:
            # Import required libraries
            from segment_anything import sam_model_registry, SamPredictor
            import numpy as np
            import cv2
            import torch
            import urllib.request
            import os

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.report({'INFO'}, f"Using device: {device}")

            # Create sam_model directory if it doesn't exist
            os.makedirs(sam_model_dir, exist_ok=True)

            # Setup SAM model path
            checkpoint_path = os.path.join(sam_model_dir, "sam_vit_b_01ec64.pth")
            model_type = "vit_b"

            # Download model if needed
            if not os.path.exists(checkpoint_path):
                self.report({'INFO'}, f"SAM model not found at {checkpoint_path}")
                self.report({'INFO'}, "Downloading SAM model (this may take a few minutes)...")
                
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                
                try:
                    def report_progress(count, block_size, total_size):
                        progress = count * block_size / total_size * 100
                        self.report({'INFO'}, f"Downloading: {progress:.1f}%")
                    
                    # Create temporary download file
                    temp_path = checkpoint_path + ".temp"
                    
                    # Download to temporary file first
                    urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)
                    
                    # Rename to final filename
                    os.rename(temp_path, checkpoint_path)
                    
                    self.report({'INFO'}, "SAM model downloaded successfully!")
                    
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to download SAM model: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return False
            else:
                self.report({'INFO'}, "Using existing SAM model checkpoint.")

            # Load and process image
            image = cv2.imread(render_path)
            if image is None:
                self.report({'ERROR'}, f"Failed to load image: {render_path}")
                return False
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]

            # Convert normalized coordinates to pixel coordinates
            try:
                pixel_points = []
                final_labels = []
                
                for point, label in zip(input_points, input_labels):
                    x = int(point[0] * W)
                    y = int((1 - point[1]) * H)  # Flip Y coordinate
                    
                    if 0 <= x < W and 0 <= y < H:
                        pixel_points.append([x, y])
                        final_labels.append(label)
                        point_type = "Positive" if label == 1 else "Negative"
                        self.report({'INFO'}, f"Added {point_type} point at ({x}, {y})")

                if not pixel_points:
                    self.report({'ERROR'}, "No valid points after conversion")
                    return False

                # Convert to numpy arrays
                input_points = np.array(pixel_points)
                input_labels = np.array(final_labels)

            except Exception as e:
                self.report({'ERROR'}, f"Point conversion error: {str(e)}")
                return False

            try:
                # Initialize SAM
                self.report({'INFO'}, f"Loading SAM model on {device.upper()}...")
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                sam.to(device=device)

                # Create predictor
                predictor = SamPredictor(sam)
                predictor.set_image(image)

                # Generate masks with multiple outputs
                self.report({'INFO'}, "Generating masks...")

                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, logits = predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True
                        )
                else:
                    masks, scores, logits = predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True
                    )

                if masks is None or len(masks) == 0:
                    self.report({'ERROR'}, "No masks generated")
                    return False

                # Select best mask based on scores
                mask_idx = np.argmax(scores)
                selected_mask = masks[mask_idx]

                # Process the mask:
                # - True (1) in the mask represents the foreground
                # - False (0) in the mask represents the background
                # We want the negative points to be black (0) in our output
                final_mask = selected_mask.copy()

                # Create a visualization for debugging
                debug_mask = np.zeros((H, W, 3), dtype=np.uint8)
                debug_mask[final_mask] = [255, 255, 255]  # White for positive regions
                
                # Draw points on debug mask
                for point, label in zip(pixel_points, final_labels):
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, Red for negative
                    cv2.circle(debug_mask, (point[0], point[1]), 5, color, -1)

                # Save debug visualization
                debug_path = mask_path.replace('_mask.png', '_debug_mask.png')
                cv2.imwrite(debug_path, debug_mask)

                # Save final mask
                self.report({'INFO'}, f"Saving mask to: {mask_path}")
                mask_saved = cv2.imwrite(mask_path, (final_mask * 255).astype(np.uint8))
                
                if not mask_saved:
                    self.report({'ERROR'}, f"Failed to save mask to {mask_path}")
                    return False

                self.report({'INFO'}, f"Successfully saved mask to {mask_path}")
                return True

            except Exception as e:
                self.report({'ERROR'}, f"SAM processing error: {str(e)}")
                return False

            finally:
                # Clean up
                if device == "cuda":
                    torch.cuda.empty_cache()
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return False

    def get_positive_points(self, material):
        """Get positive points from the material's prompts"""
        return material.prompts.positive_prompts

    def get_negative_points(self, material):
        """Get negative points from the material's prompts"""
        return material.prompts.negative_prompts

    def load_image(self, image_path):
        # Load the image using PIL or any other library
        from PIL import Image
        return Image.open(image_path)

    def create_mask(self, image, positive_points, negative_points):
        # Create a mask image
        mask = Image.new("L", image.size, 0)  # Create a black mask

        # Set positive points to white
        for point in positive_points:
            mask.putpixel((int(point[0]), int(point[1])), 255)  # White for positive points

        # Set negative points to black (already black, but for clarity)
        for point in negative_points:
            mask.putpixel((int(point[0]), int(point[1])), 0)  # Black for negative points

        return mask

# Registration
classes = (
    OBJECT_OT_Select3DObject,
    OBJECT_OT_AddMaterialID,
    OBJECT_OT_DeleteMaterialID,
    OBJECT_OT_AddCamera,
    OBJECT_OT_DeleteCamera,
    OBJECT_OT_AddPositivePoint,
    OBJECT_OT_AddNegativePoint,
    OBJECT_OT_RenderMask,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()