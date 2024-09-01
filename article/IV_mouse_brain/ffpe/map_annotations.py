import warnings
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from paquo.projects import QuPathProject
from shapely.errors import ShapelyDeprecationWarning


def get_annotation_polygons(qupath_project, show=True):
    # Filter out ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
    # load the qupath project
    slides = QuPathProject(qupath_project, mode='r+')
    img_dict = {}
    for img in slides.images:

        # get annotations for slide image
        annotations = img.hierarchy.annotations

        # collect polys
        polys = {}
        error_count = 0
        erroneous_annot = {}
        for annotation in annotations:
            try:
                id = annotation.path_class.id
                if id in polys.keys():
                    if annotation.roi.type != 'LineString':
                        polys[id].append(annotation.roi)
                else:
                    if annotation.roi.type != 'LineString':
                        polys[id] = [annotation.roi]
            except Exception:
                erroneous_annot[error_count] = annotation
                error_count += 1
        print(f"Reading slide {img.image_name}")
        print(f"Erroneous poly found {error_count} times from {len(annotations)} polygons.")

        if show:
            # merge polys with the same annotation
            polym = {}
            for key in polys.keys():
                polym[key] = unary_union(polys[key])
            # look at them
            for key in polym.keys():
                if polym[key].type != 'Polygon':
                    for geom in polym[key].geoms:
                        plt.plot(*geom.exterior.xy)
                else:
                    plt.plot(*polym[key].exterior.xy)
            plt.show()

        img_dict[img.image_name] = polys
    return img_dict


def map_annotations(adata, polygon_dict, default_annot='Tumor'):
    df_dict = {i: default_annot for i in list(adata.obs.index)}
    tissue_type = pd.DataFrame(df_dict.values(), index=df_dict.keys())
    spot_df = pd.DataFrame(adata.obsm['spatial'])

    spot_annots = {}
    for key in tqdm(polygon_dict.keys(), desc='Mapping annotations...'):
        x, y = spot_df.iloc[:, 0], spot_df.iloc[:, 1]
        points = [Point(x, y) for x, y in zip(x, y)]

        contains_sum = [False for x in range(len(spot_df))]
        for iv in polygon_dict[key]:
            contains = [iv.contains(p) for p in points]
            contains_sum = [cs or c for cs, c in zip(contains_sum, contains)]
            # plt.scatter(x, y, s=1)
            # plt.plot(*iv.exterior.xy)
            # plt.show()

        spot_annots[key] = contains_sum
        replace = adata.obs.index[contains_sum]

        tissue_type[0][replace] = key
    return tissue_type



data_path = 'data/Visium_FFPE_Mouse_Brain/'
adata = sc.read_h5ad(data_path + 'chr_28.h5ad')

qupath_project = 'data/Visium_FFPE_Mouse_Brain/smooth_brain/'

polygon_dict = get_annotation_polygons(qupath_project, show=True)

img_key = list(polygon_dict.keys())[0]
polygon_dict = polygon_dict[img_key]
annots = map_annotations(adata, polygon_dict, default_annot='Rest')
annots[0] = ['Corpus callosum' if x == 'Corpus_callosum/capsula_externa' else x for x in annots[0]]
adata.obs['annotation'] = annots

sc.pl.spatial(adata, color='annotation')
plt.show()

adata.write_h5ad(data_path + 'chr_28_annotated.h5ad')
