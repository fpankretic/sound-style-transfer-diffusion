import torch
import torch.nn.functional as F

import laion_clap

def content_preservation(content_file_list, stylized_content_file_list):
    """
    Parameters:
        content_file_list (list[str]): list of content file (.wav format) locations, must match stylized_content_file_list (pairs)
        stylized_content_file_list (list[str]): list of stylized content file (.wav format) locations, must match content_file_list (pairs)

    Returns:
        content_preservation (float): content preservation metric calculated for input file lists
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
    
    return avg_content_preservation

def style_fit(style_file_list, stylized_content_file_list):
    """
    Parameters:
        style_file_list (list[str]): list of style file (.wav format) locations (style files used for training the model)
        stylized_content_file_list (list[str]): list of stylized content file (.wav format) locations

    Returns:
        style_fit (float): style fit metric calculated for input file list
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
    
    return avg_style_fit

if __name__ == '__main__':
    # Content preservation test
    accordion_list_1 = ['./audios/timbre/accordion/accordion1.wav', './audios/timbre/accordion/accordion2.wav', './audios/timbre/accordion/accordion3.wav']
    accordion_list_2 = ['./audios/timbre/accordion/accordion4.wav', './audios/timbre/accordion/accordion5.wav', './audios/timbre/accordion/accordion6.wav']

    print(content_preservation(accordion_list_1, accordion_list_2))

    # Style fit test
    accordion_style = ['./audios/timbre/accordion/accordion7.wav']

    print(style_fit(accordion_style, accordion_list_1))