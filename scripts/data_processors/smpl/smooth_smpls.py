
import os
import pathlib
import bpy
import os
from os.path import join
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler
import sys

this_script_path = pathlib.Path(__file__).parent.resolve()
#starts at 0
CHARACTER = 0
SMPL_TEMPLATE_PATH = 'basicModel_m_lbs_10_207_0_v1.0.2.fbx'
SMPL_TEMPLATE_PATH = this_script_path / "blend" / "basicModel_m_lbs_10_207_0_v1.0.2.fbx"
ARM_OBJ_NAME = "Finalized_Armature"
OBJ_NAME = "Finalized_Mesh"
PART_MATCH_DICT = {'root': 'root', 'bone_00':  'Pelvis', 'bone_01':  'L_Hip', 'bone_02':  'R_Hip', 
                    'bone_03':  'Spine1', 'bone_04':  'L_Knee', 'bone_05':  'R_Knee', 'bone_06':  'Spine2', 
                    'bone_07':  'L_Ankle', 'bone_08':  'R_Ankle', 'bone_09':  'Spine3', 'bone_10':  'L_Foot', 
                    'bone_11':  'R_Foot', 'bone_12':  'Neck', 'bone_13':  'L_Collar', 'bone_14':  'R_Collar', 
                    'bone_15':  'Head', 'bone_16':  'L_Shoulder', 'bone_17':  'R_Shoulder', 'bone_18':  'L_Elbow', 
                    'bone_19':  'R_Elbow', 'bone_20':  'L_Wrist', 'bone_21':  'R_Wrist',
                    'bone_22':  'L_Hand', 'bone_23':  'R_Hand',
                    
                    }

def rodrigues2bshapes(body_pose):
    mat_rots = body_pose
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                            for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)

# apply trans pose and shape to character
def apply_trans_pose_shape(trans, body_pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):

    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(body_pose)

    part_bones  = PART_MATCH_DICT
    arm_ob.pose.bones['m_avg_Pelvis'].location = trans
    arm_ob.pose.bones['m_avg_Pelvis'].keyframe_insert('location', frame=frame)
    
    arm_ob.pose.bones['m_avg_root'].rotation_quaternion.w = 0.0
    arm_ob.pose.bones['m_avg_root'].rotation_quaternion.x = -1.0

    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_bones['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert(
                'value', index=-1, frame=frame)

def init_scene():
    path_fbx = SMPL_TEMPLATE_PATH
    bpy.ops.import_scene.fbx(filepath=str(path_fbx.absolute()), axis_forward='-Y', axis_up='-Z', global_scale=100)

    obj_gender = 'm'
    obname = '%s_avg' % obj_gender
    ob = bpy.data.objects[obname]
    arm_obj = 'Armature'

    print('success load')
    
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = ''
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects[arm_obj]
    arm_ob.animation_data_clear()
    
    return(ob, obname, arm_ob, cam_ob)

def import_smpls_group(smpls_group_path):
    results = np.load(smpls_group_path, allow_pickle=True)
    params = []
    object_name = 'm_avg'
    obj_gender = 'm'
    scene = bpy.data.scenes['Scene']
    ob, obname, arm_ob, cam_ob= init_scene()

    obj = bpy.context.window.scene.objects[object_name]
    bpy.context.view_layer.objects.active = ob

    obs = []
    for ob in bpy.context.scene.objects:
        if ob.type == 'ARMATURE':
            obs.append(ob)

    obs[len(obs)-1].select_set(True)
    view_layer = bpy.context.view_layer
    view_layer.objects.active = arm_ob
    scene.frame_end = len(results["smpl"])
    scene.frame_start = 0
    for fframe, data in enumerate(zip(results["smpl"],results["camera"])):
        #print('characters_index max:',len(data[0])-1)
        if CHARACTER <= len(data[0])-1:
            scene.frame_set(fframe)
            trans = data[1]
            trans[1] *= -1
            shape = data[0]['betas'][CHARACTER]
            global_orient = data[0]['global_orient'][CHARACTER].reshape((-1, 3, 3))
            body_pose = data[0]['body_pose'][CHARACTER].reshape((-1, 3, 3))
            final_body_pose = np.vstack([global_orient, body_pose])
            apply_trans_pose_shape(Vector(trans), final_body_pose, shape, obj, arm_ob, obname, scene, cam_ob, fframe)
            bpy.context.view_layer.update()
        else:
            print('skipping to the next')
        
    arm_ob.name = ARM_OBJ_NAME
    obj.name=OBJ_NAME
    print(f"Import SMPLs Sequence from:{smpls_group_path}.")
    
def reverse_blender_para_to_smpl(ob, arm_ob, obname, scene, frame=None):
    scene.frame_set(frame)
    trans = arm_ob.pose.bones['m_avg_Pelvis'].location 
    part_bones  = PART_MATCH_DICT
    mrots = np.zeros((24,3,3))
    shape = np.zeros((10,))
    for ibshape, shape_elem in enumerate(shape):
        shape[ibshape] = ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value
        
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_bones['bone_%02d' % ibone]]
        mrots[ibone] = np.array(bone.rotation_quaternion.to_matrix().to_3x3())
    
    smpl_result = {"global_orient":mrots[0].reshape((1,-1,3,3)), "betas":shape.reshape((1,-1)), "body_pose":mrots[1:].reshape((1,-1,3,3))}
    cam_result = np.array(trans)
    return smpl_result, cam_result

def export_smpls_group(smpls_save_path):
    obj = bpy.data.objects[OBJ_NAME]
    arm_ob = bpy.data.objects[ARM_OBJ_NAME]
    scene = bpy.data.scenes['Scene']
    obname = "m_avg"
    
    smpls = []
    cams = []
    for frame in range(scene.frame_start, scene.frame_end):
        smpl_result, cam_result = reverse_blender_para_to_smpl(obj, arm_ob, obname, bpy.data.scenes['Scene'], frame=frame)
        smpls.append(smpl_result)
        cams.append(cam_result)
    
    np.savez(smpls_save_path, smpl=smpls, camera=cams)
    print(f"Save SMPLs to: {smpls_save_path}")

def smooth(arm_ob_name=ARM_OBJ_NAME):
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end
    bpy.ops.nla.bake(frame_start=start_frame, frame_end=end_frame, 
                     only_selected=False, visual_keying=True, clear_constraints=False, 
                     clear_parents=False, use_current_action=False, clean_curves=False, bake_types={'POSE'})
    
    def smooth_curves(o):
        layer = bpy.context.view_layer
        layer.objects.active = o
        # select all (relevant) bones
        for b in o.data.bones:
            b.select = True
        layer.update()
        
        bpy.context.window.screen.areas[0].type = 'GRAPH_EDITOR'
        area_type = 'GRAPH_EDITOR' # change this to use the correct Area Type context you want to process in
        areas  = [area for area in bpy.context.window.screen.areas if area.type == area_type]
        
        if len(areas) <= 0:
            raise Exception(f"Make sure an Area of type {area_type} is open or visible in your screen!")

        with bpy.context.temp_override(
            window=bpy.context.window,
            area=areas[0],
            region=[region for region in areas[0].regions if region.type == 'WINDOW'][0],
            screen=bpy.context.window.screen
        ):
            layer.update()
            bpy.ops.graph.smooth()

    # currently selected 
    arm_ob = bpy.data.objects[arm_ob_name]
    smooth_curves(arm_ob)
    print('SMOOTH FINISHED!')
    
    
if __name__ == "__main__":
    argv = sys.argv
    smpls_path = argv[argv.index("--smpls_group_path") + 1]
    smpls_smoothed_path = argv[argv.index("--smoothed_result_path") + 1]

    import_smpls_group(smpls_path)
    smooth()
    export_smpls_group(smpls_smoothed_path)
    

    


    

