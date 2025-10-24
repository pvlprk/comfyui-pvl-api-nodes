import torch
import torch.nn.functional as F

MAX_RESOLUTION = 8192  # adjust as needed


class PVL_ImageComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "offset_x": ("STRING", {"multiline": False, "default": "0"}),
                "offset_y": ("STRING", {"multiline": False, "default": "0"}),
                "delimiter": ("STRING", {"multiline": False, "default": "[++]"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "pvl/image manipulation"

    # -------------------------------------------------------
    def _parse_offset_list(self, offset_str, delimiter, batch_len):
        """Parse a string of numbers separated by a delimiter into a list of floats, filling to match batch_len."""
        try:
            values = [float(v.strip()) for v in offset_str.split(delimiter) if v.strip() != ""]
        except Exception:
            values = [0.0]
        if not values:
            values = [0.0]
        if len(values) < batch_len:
            values += [values[-1]] * (batch_len - len(values))
        return values[:batch_len]

    # -------------------------------------------------------
    def _match_batch_count(self, tensor_a, tensor_b):
        """Repeat or truncate tensors so both have same batch dimension."""
        if tensor_a.shape[0] == tensor_b.shape[0]:
            return tensor_a, tensor_b
        elif tensor_a.shape[0] == 1:
            tensor_a = tensor_a.repeat(tensor_b.shape[0], 1, 1, 1)
            return tensor_a, tensor_b
        elif tensor_b.shape[0] == 1:
            tensor_b = tensor_b.repeat(tensor_a.shape[0], 1, 1, 1)
            return tensor_a, tensor_b
        else:
            # Unequal >1 counts → pad smaller one by repeating last
            max_b = max(tensor_a.shape[0], tensor_b.shape[0])
            if tensor_a.shape[0] < max_b:
                tensor_a = torch.cat((tensor_a, tensor_a[-1:].repeat(max_b - tensor_a.shape[0], 1, 1, 1)), dim=0)
            if tensor_b.shape[0] < max_b:
                tensor_b = torch.cat((tensor_b, tensor_b[-1:].repeat(max_b - tensor_b.shape[0], 1, 1, 1)), dim=0)
            return tensor_a, tensor_b

    # -------------------------------------------------------
    def execute(self, destination, source, x, y, offset_x, offset_y, delimiter="[++]", mask=None):
        # Align batch counts for destination/source first
        destination, source = self._match_batch_count(destination, source)
        batch_len = max(destination.shape[0], source.shape[0])

        # Match / replicate mask if present
        if mask is None:
            mask = torch.ones_like(source)[:, :, :, 0]
        if mask.ndim == 3:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        if mask.shape[1:3] != source.shape[1:3]:
            mask = F.interpolate(
                mask.permute([0, 3, 1, 2]),
                size=(source.shape[1], source.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).permute([0, 2, 3, 1])
        # Batch match
        if mask.shape[0] != batch_len:
            if mask.shape[0] == 1:
                mask = mask.repeat(batch_len, 1, 1, 1)
            elif mask.shape[0] < batch_len:
                mask = torch.cat((mask, mask[-1:].repeat(batch_len - mask.shape[0], 1, 1, 1)), dim=0)
            else:
                mask = mask[:batch_len]

        # Parse offset strings
        offset_x_list = self._parse_offset_list(offset_x, delimiter, batch_len)
        offset_y_list = self._parse_offset_list(offset_y, delimiter, batch_len)

        # Static integer x/y reused for all
        x_list = [int(x)] * batch_len
        y_list = [int(y)] * batch_len

        # Adjusted coords
        x_list = [x_list[i] + int(offset_x_list[i]) for i in range(batch_len)]
        y_list = [y_list[i] + int(offset_y_list[i]) for i in range(batch_len)]

        # Now composite per image
        output = []
        for i in range(batch_len):
            d = destination[min(i, destination.shape[0] - 1)].clone()
            s = source[min(i, source.shape[0] - 1)]
            m = mask[min(i, mask.shape[0] - 1)]

            yi, xi = y_list[i], x_list[i]

            # Clamp crop if source goes out of bounds
            sH, sW = s.shape[0], s.shape[1]
            dH, dW = d.shape[0], d.shape[1]

            cropH = max(0, min(sH, dH - yi))
            cropW = max(0, min(sW, dW - xi))
            if cropH <= 0 or cropW <= 0:
                output.append(d)
                continue

            s_c = s[:cropH, :cropW, :]
            m_c = m[:cropH, :cropW, :]

            d[yi:yi+cropH, xi:xi+cropW, :] = s_c * m_c + d[yi:yi+cropH, xi:xi+cropW, :] * (1 - m_c)
            output.append(d)

        return (torch.stack(output),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PVL_ImageComposite": PVL_ImageComposite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PVL_ImageComposite": "PVL Image Composite"
}
