import torch


def unpack_node_params(node_params, num_coords, num_velocities):
    if num_coords > 0 and num_velocities > 0:
        coords = node_params[:, :num_coords]
        velocities = node_params[:, num_coords: num_coords + num_velocities]
        properties = node_params[:, num_coords + num_velocities:]
        if num_coords + num_velocities == node_params.size()[1]:
            properties = torch.cat([properties, torch.sum(velocities ** 2, 1).unsqueeze(1)], dim=1)
    elif num_coords > 0 and num_velocities == 0:
        coords = node_params[:, :num_coords]
        velocities = None
        properties = node_params[:, num_coords:]
    else:
        coords = None
        velocities = None
        properties = node_params
    return coords, velocities, properties


def pack_node_params(coords, velocities, properties):
    res = properties
    if velocities is not None:
        res = torch.cat([velocities, res], dim=1)
    if coords is not None:
        res = torch.cat([coords, res], dim=1)
    return res


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
