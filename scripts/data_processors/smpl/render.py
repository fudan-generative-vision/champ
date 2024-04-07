import pip

pip.main(["install", "pandas"])
pip.main(["install", "tqdm"])

import bpy
import numpy as np
import pandas as pd
from pathlib import Path
import os
from ast import literal_eval
from tqdm import tqdm
from multiprocessing import Pool
from contextlib import contextmanager
import pathlib

this_script_path = pathlib.Path(__file__).parent.resolve()
W_FACE_AND_COLOR_FILE = this_script_path / "blend/smpl_mesh_info.npy"

FORMAT_LDR = "PNG"
COLOR_DEPTH_LDR = 8
SAMPLES = 1
COLOR_MODE = "RGB"


def setup_device(use_id):
    bpy.context.scene.render.engine = "CYCLES"

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

    for i, device in enumerate(
        bpy.context.preferences.addons["cycles"].preferences.devices
    ):
        if i == use_id or "CPU" in device["name"]:
            device["use"] = True  # Using all devices, include GPU and CPU
        else:
            device["use"] = False  # Using all devices, include GPU and CPU
        print(device["name"], "USE:", bool(device["use"]))


class SingleDataset:
    def __init__(self, smpl_folder, smpl_suffixes=["npy", "npz"]):
        self.smpl_folder = Path(smpl_folder)
        self.out_folder = Path(smpl_folder).parent
        self.smpl_paths = []
        self.bboxes = []
        self.valid_index = []

        folder = self.out_folder
        smpl_path_this_forlder = sorted(
            list(
                [
                    path
                    for i in smpl_suffixes
                    for path in (folder / "smpl_results").glob("*." + i)
                ]
            )
        )
        self.smpl_paths += smpl_path_this_forlder

        self.valid_index += [True] * len(smpl_path_this_forlder)

        self.smpl_paths = [
            path for path, valid in zip(self.smpl_paths, self.valid_index) if valid
        ]
        self.output_paths = [self.out_folder for smpl_path in self.smpl_paths]
        # Skip finished smpl_path. Enable it if want to continue processing only remaining imgs.
        smpl_fns = [
            os.path.splitext(os.path.basename(smpl_path))[0]
            for smpl_path in self.smpl_paths
        ]  # Example smpl_fn:0000_all
        imgs_output_path = [
            os.path.join(str(self.output_paths[i]), "visualized_imgs", f"{smpl_fn}.png")
            for i, smpl_fn in enumerate(smpl_fns)
        ]
        imgs_already_exist = [
            os.path.exists(smpl_path) for smpl_path in imgs_output_path
        ]
        imgs_index_to_inference = np.where(np.array(imgs_already_exist) == False)[0]
        smpl_paths_copy = list(self.smpl_paths)
        output_paths_copy = list(self.output_paths)
        self.smpl_paths = [
            smpl_paths_copy[img_index] for img_index in imgs_index_to_inference
        ]
        self.output_paths = [
            output_paths_copy[img_index] for img_index in imgs_index_to_inference
        ]
        print(
            f"finish loading {imgs_index_to_inference.shape[0]} frames data, skip {len(smpl_fns) - imgs_index_to_inference.shape[0]} existing images"
        )


def load_smpl(smpl_path):
    return np.load(smpl_path, allow_pickle=True).item()


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def rendering_pipeline(dataset_totol, ref_img_path):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = str(FORMAT_LDR)
    scene.render.image_settings.color_depth = str(COLOR_DEPTH_LDR)
    scene.render.image_settings.color_mode = str(COLOR_MODE)
    scene.render.resolution_percentage = 100
    scene.render.use_persistent_data = True
    scene.cycles.use_denoising = False

    camera = bpy.data.objects["Camera"]
    camera.data.clip_start = 0.05
    camera.data.clip_end = 1e12
    camera.data.cycles.samples = SAMPLES
    scene.cycles.samples = SAMPLES
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    render_layers = bpy.context.scene.view_layers
    if "mesh_collection" not in bpy.data.collections.keys():
        mesh_collection = bpy.data.collections.new("mesh_collection")
        bpy.context.scene.collection.children.link(mesh_collection)
    mesh_collection = bpy.data.collections.get("mesh_collection")
    mat_semantic = bpy.data.materials.get("Semantic")
    mat_normal = bpy.data.materials.get("Normal")

    result_dict = np.load(W_FACE_AND_COLOR_FILE, allow_pickle=True).item()
    faces = result_dict["faces"]
    verts_color = result_dict["verts_color"][:, ::-1].astype(np.float32) / 255.0

    # dataset_sections = np.array_split(np.arange(len(dataset_totol.smpl_paths)),int(len(dataset_totol.smpl_paths)/100)+1)

    result_dict_list = []
    # smpl_paths_batch = [dataset_totol.smpl_paths[i] for i in section]
    # out_paths_batch = [dataset_totol.output_paths[i] for i in section]
    with Pool(4) as p:
        processed = p.map(
            load_smpl,
            tqdm(
                dataset_totol.smpl_paths,
                total=len(dataset_totol.smpl_paths),
                desc="Loading smpls into RAM",
            ),
        )

    for smpl in tqdm(
        processed, total=len(dataset_totol.smpl_paths), desc="Loading smpls into RAM"
    ):
        result_dict_list.append(smpl)
    # for smpl_path, output_path in tqdm(zip(dataset_totol.smpl_paths, dataset_totol.output_paths), total=len(dataset_totol.smpl_paths), desc ="Loading smpls into RAM"):
    #     result_dict = np.load(smpl_path, allow_pickle=True).item()
    #     result_dict_list.append(result_dict)

    for smpl_path, output_path, result_dict in tqdm(
        zip(dataset_totol.smpl_paths, dataset_totol.output_paths, result_dict_list),
        total=len(dataset_totol.output_paths),
        desc="Rendering Images",
        miniters=10,
    ):
        render_path = output_path
        smpl_fn, _ = os.path.splitext(os.path.basename(smpl_path))
        smpl_fn = smpl_fn.split(".")[0]
        # bpy.ops.image.open(filepath="C:/Users/Leo/Desktop/smpl_rendering/454796.png", directory="C:\\Users\\Leo\\Desktop\\smpl_rendering\\",
        #    files=[{"name":"ori_img.png", "name":"ori_img.png"}], show_multiview=False)
        ori_img = bpy.data.images.load(ref_img_path)
        ori_img.name = "ori_img.png"
        ori_img.colorspace_settings.name = "Raw"
        bpy.data.scenes["Scene"].node_tree.nodes["Image"].image = ori_img

        cam_t = result_dict["cam_t"][0]
        verts = result_dict["verts"][0] + cam_t
        img_size = result_dict["render_res"].astype(np.int32)
        # img_size[1] += 200
        smpl_outs = result_dict["smpls"]
        camera.data.sensor_width = img_size.max()
        camera.data.lens = result_dict["scaled_focal_length"]
        scene.render.resolution_x = img_size[0]
        scene.render.resolution_y = img_size[1]

        # make mesh
        new_mesh = bpy.data.meshes.new("smpl_mesh")
        new_mesh.from_pydata(verts, edges=[], faces=faces)
        #    new_mesh.shade_smooth()
        # make object from mesh
        new_object = bpy.data.objects.new("new_object", new_mesh)
        # make collection
        new_object.rotation_euler[0] = -np.pi / 2

        for f in new_object.data.polygons:
            f.use_smooth = True

        # add object to scene collection
        mesh_collection.objects.link(new_object)

        # Render Normal Map and Depth Map
        bpy.data.scenes["Scene"].node_tree.nodes["Depth Output"].base_path = (
            os.path.join(render_path, "depth")
        )
        bpy.data.scenes["Scene"].node_tree.nodes["Visualize Output"].base_path = (
            os.path.join(render_path, "visualized_imgs")
        )
        bpy.data.scenes["Scene"].node_tree.nodes["Mask Output"].base_path = (
            os.path.join(render_path, "mask")
        )

        new_object.data.materials.append(mat_normal)

        scene.view_settings.view_transform = "Raw"
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            0
        ].default_value = (0, 0, 0, 1)
        output_name = f"{smpl_fn}.png"
        depth_path = os.path.join(render_path, "depth", output_name)
        normal_path = os.path.join(render_path, "normal", output_name)
        vis_path = os.path.join(render_path, "visualized_imgs", output_name)
        mask_path = os.path.join(render_path, "mask", output_name)
        semantic_path = os.path.join(render_path, "semantic_map", output_name)
        scene.render.filepath = normal_path
        for layer in render_layers:
            # some condition
            layer.use = layer.name == "ViewLayer"
        bpy.context.scene.render.film_transparent = True
        with stdout_redirected():
            bpy.ops.render.render(write_still=True)

        if os.path.isfile(depth_path):
            os.remove(depth_path)
        if os.path.isfile(os.path.join(render_path, "depth", f"{0:04d}.png")):
            os.rename(os.path.join(render_path, "depth", f"{0:04d}.png"), depth_path)

        if os.path.isfile(vis_path):
            os.remove(vis_path)
        if os.path.isfile(os.path.join(render_path, "visualized_imgs", f"{0:04d}.png")):
            os.rename(
                os.path.join(render_path, "visualized_imgs", f"{0:04d}.png"), vis_path
            )

        if os.path.isfile(mask_path):
            os.remove(mask_path)
        if os.path.isfile(os.path.join(render_path, "mask", f"{0:04d}.png")):
            os.rename(os.path.join(render_path, "mask", f"{0:04d}.png"), mask_path)

        # This is to reference the vertex color layer later
        vertex_colors_name = "vert_colors"
        # Here the color layer is made on the mesh
        new_mesh.vertex_colors.new(name=vertex_colors_name)
        # We define a variable that is used to easily reference
        # the color layer in code later
        color_layer = new_mesh.vertex_colors[vertex_colors_name]

        # We loop over all the polygons
        for poly in new_mesh.polygons:
            # We get the polygon index and the corresponding mesh index
            for vert_i_poly, vert_i_mesh in enumerate(poly.vertices):
                # We get the loop index from the polygon index
                vert_i_loop = poly.loop_indices[vert_i_poly]
                # We set the color for the vertex
                rgb = verts_color[vert_i_mesh].tolist()
                rgb.append(1)
                color_layer.data[vert_i_loop].color = rgb
        del rgb
        mat_semantic.node_tree.nodes["Color Attribute"].layer_name = "vert_colors"
        new_object.data.materials[0] = mat_semantic
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            0
        ].default_value = (0.05781, 0.0003, 0.08866, 1)
        scene.view_settings.view_transform = "Standard"
        scene.render.filepath = semantic_path

        for layer in render_layers:
            # some condition
            layer.use = layer.name == "VertexColor Layer"
        bpy.context.scene.render.film_transparent = False
        with stdout_redirected():
            bpy.ops.render.render(write_still=True)

        # objs = [bpy.context.scene.objects['new_object']]
        # with bpy.context.temp_override(selected_objects=objs):
        #     bpy.ops.object.delete()
        bpy.data.objects.remove(new_object, do_unlink=True)
        bpy.data.meshes.remove(new_mesh)
        bpy.data.images.remove(ori_img)
        if os.path.isfile(os.path.join(render_path, "visualized_imgs", f"{0:04d}.png")):
            os.remove(os.path.join(render_path, "visualized_imgs", f"{0:04d}.png"))
        del new_mesh
        del ori_img
        del new_object
    del result_dict_list


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # rendering_pipeline(HumansDataset(SMPL_FOLDER, OUT_FOLDER))
    import sys

    argv = sys.argv
    print(f"Rendering:")
    try:
        argv.index("--device")
    except:
        print("Use Only CPU for Rendering")
    else:
        setup_device(int(argv[argv.index("--device") + 1]))

    smpl_folder = argv[argv.index("--driving_path") + 1]
    ref_img_path = argv[argv.index("--reference_path") + 1]
    rendering_pipeline(SingleDataset(smpl_folder), ref_img_path)
