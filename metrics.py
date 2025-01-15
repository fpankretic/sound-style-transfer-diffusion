import torch
import torch.nn.functional as F

import laion_clap

@torch.no_grad()
def content_preservation(content_file_list, stylized_content_file_list):
    """
    Parameters:
        content_file_list (list[str]): list of content file (.wav format) locations, must match stylized_content_file_list (pairs)
        stylized_content_file_list (list[str]): list of stylized content file (.wav format) locations, must match content_file_list (pairs)

    Returns:
        content_preservation (float): content preservation metric calculated for input file lists
        best_index (int): index of pair with best content preservation
        best_val (float): best content preservation
    """
    # Load model
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(verbose=False)

    # Get audio embeddings from audio files
    audio_embeddings_content = model.get_audio_embedding_from_filelist(x=content_file_list, use_tensor=True)
    audio_embeddings_stylized_content = model.get_audio_embedding_from_filelist(x=stylized_content_file_list, use_tensor=True)

    # Calculate content preservation
    cos_sim = F.cosine_similarity(audio_embeddings_content, audio_embeddings_stylized_content, dim=1)
    content_preservation = torch.maximum(cos_sim, torch.tensor(0))

    # Calculate average content preservation
    avg_content_preservation = torch.mean(content_preservation).item()

    # Find index and value of pair with best content preservation
    best_index = torch.argmax(content_preservation).item()
    best_val = content_preservation[best_index].item()
    
    return avg_content_preservation, best_index, best_val

@torch.no_grad()
def style_fit(style_file_list, stylized_content_file_list):
    """
    Parameters:
        style_file_list (list[str]): list of style file (.wav format) locations (style files used for training the model)
        stylized_content_file_list (list[str]): list of stylized content file (.wav format) locations

    Returns:
        style_fit (float): style fit metric calculated for input file list
        best_index (int): index of pair with best style fit
        best_val (float): best style fit
    """
    # Load model
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(verbose=False)

    # Get audio embeddings from audio files
    audio_embedding_style = torch.mean(model.get_audio_embedding_from_filelist(x=style_file_list, use_tensor=True), dim=0).reshape(1, -1)
    audio_embeddings_stylized_content = model.get_audio_embedding_from_filelist(x=stylized_content_file_list, use_tensor=True)

    # Calculate style fit
    cos_sim = F.cosine_similarity(audio_embedding_style, audio_embeddings_stylized_content, dim=1)
    style_fit = torch.maximum(cos_sim, torch.tensor(0))

    # Calculate average content preservation
    avg_style_fit = torch.mean(style_fit).item()

    # Find index and value of pair with best style fit
    best_index = torch.argmax(style_fit).item()
    best_val = style_fit[best_index].item()
    
    return avg_style_fit, best_index, best_val

if __name__ == '__main__':
    # Content preservation test
    content = ['./audios/comparison/hiphop1.wav']
    stylized_content = ['./audios/comparison/out_sample_1.wav', './audios/comparison/out_sample_2.wav']

    print(f'Content preservation: {content_preservation(content, stylized_content)}')

    # Style fit test
    style = ['./audios/comparison/cornet1.wav']

    print(f'Style fit: {style_fit(style, stylized_content)}')