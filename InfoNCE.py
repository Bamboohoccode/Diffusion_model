import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. HÀM MẤT MÁT INFONCE (InfoNCE Loss)
# Chuẩn theo bài viết: "InfoNCE: Explained in Details and Implementations"
# =====================================================================
class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Contrastive Predictive Coding / SimCLR / CLIP style)
    
    Công thức toán học:
        L = - E [ log ( exp(sim(q, k+) / tau) / (exp(sim(q, k+) / tau) + sum_j exp(sim(q, k-_j) / tau)) ) ]
    
    Tham số:
        - temperature (tau): Hệ số nhiệt độ điều chỉnh độ phân tách của softmax 
                             (thường từ 0.05 đến 0.2, mặc định 0.1).
        - reduction: 'mean' hoặc 'sum'
    """
    def __init__(self, temperature: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query: torch.Tensor, positive_key: torch.Tensor, negative_keys: torch.Tensor = None):
        """
        Đầu vào:
            query: Tensor kích thước [Batch_size, Dim]
            positive_key: Tensor kích thước [Batch_size, Dim] (mỗi query có 1 positive tương ứng)
            negative_keys: (Tùy chọn)
                - Nếu None: Chế độ "In-batch negatives" (SimCLR/CLIP).
                            Các mẫu khác trong cùng batch sẽ tự động đóng vai trò là negative.
                - Nếu có: Tensor kích thước [Num_negatives, Dim] hoặc [Batch_size, Num_negatives, Dim].
        """
        # Bước 1: Chuẩn hóa L2 vector đặc trưng (Cosine Similarity = Dot product sau khi normalize)
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)

        # Tính độ tương đồng giữa query và positive key: shape [Batch_size, 1]
        pos_sim = torch.sum(query * positive_key, dim=-1, keepdim=True) / self.temperature

        if negative_keys is None:
            # =========================================================
            # Chế độ 1: In-batch Negatives (như trong SimCLR, CLIP)
            # =========================================================
            # Ma trận tương đồng giữa tất cả query và tất cả positive_key trong batch:
            # shape: [Batch_size, Batch_size]
            all_sim = torch.matmul(query, positive_key.T) / self.temperature

            # Trên đường chéo chính (i == j) là các cặp (query_i, positive_i)
            # Các phần tử ngoài đường chéo chính (i != j) tự động là các negative keys!
            labels = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
            loss = F.cross_entropy(all_sim, labels, reduction=self.reduction)

        else:
            # =========================================================
            # Chế độ 2: Explicit Negatives (Mẫu âm được chỉ định trực tiếp)
            # =========================================================
            negative_keys = F.normalize(negative_keys, dim=-1)

            if negative_keys.dim() == 2:
                # Trường hợp tập negatives dùng chung cho cả batch: [Num_negatives, Dim]
                # neg_sim có shape: [Batch_size, Num_negatives]
                neg_sim = torch.matmul(query, negative_keys.T) / self.temperature
            elif negative_keys.dim() == 3:
                # Trường hợp mỗi sample có tập negatives riêng: [Batch_size, Num_negatives, Dim]
                # neg_sim có shape: [Batch_size, Num_negatives]
                neg_sim = torch.bmm(negative_keys, query.unsqueeze(-1)).squeeze(-1) / self.temperature
            else:
                raise ValueError("negative_keys phải có 2 hoặc 3 chiều.")

            # Ghép positive và negatives vào chung một ma trận so khớp:
            # Cột đầu tiên (index 0) là positive, các cột sau (1 -> K) là negatives
            # logits có shape: [Batch_size, 1 + Num_negatives]
            logits = torch.cat([pos_sim, neg_sim], dim=-1)

            # Vì positive luôn nằm ở cột 0 nên target là vector toàn số 0
            labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


# =====================================================================
# 2. TOY DEMO: HUẤN LUYỆN CONTRASTIVE LEARNING VỚI INFONCE
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== DEMO 1: KIỂM TRA CHỨC NĂNG CỦA INFONCE LOSS ===")
    
    batch_size = 8
    feature_dim = 64
    temperature = 0.07

    info_nce = InfoNCELoss(temperature=temperature)

    # Giả lập embedding của Query và Positive
    query = torch.randn(batch_size, feature_dim)
    # Positive tương đồng với Query (cộng một ít nhiễu nhỏ)
    positive = query + 0.1 * torch.randn(batch_size, feature_dim)

    # 1. Chế độ In-batch negatives (tự so sánh với 7 mẫu còn lại trong batch)
    loss_inbatch = info_nce(query, positive)
    print(f"1. In-batch Negatives Loss: {loss_inbatch.item():.4f}")

    # 2. Chế độ Explicit Negatives (mỗi sample có K = 10 negatives riêng biệt)
    K = 10
    explicit_negatives = torch.randn(batch_size, K, feature_dim)
    loss_explicit = info_nce(query, positive, explicit_negatives)
    print(f"2. Explicit Negatives Loss (K={K}): {loss_explicit.item():.4f}")


    # =================================================================
    # DEMO 2: HUẤN LUYỆN 1 MẠNG ENCODER HỌC GOM CỤM DỮ LIỆU BẰNG INFONCE
    # =================================================================
    print("\n=== DEMO 2: HUẤN LUYỆN ENCODER VỚI INFONCE ===")
    
    # Mạng Encoder ánh xạ dữ liệu đầu vào (ví dụ 10 chiều) sang biểu diễn (32 chiều)
    encoder = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    for epoch in range(1, 201):
        # Giả lập dữ liệu gốc X
        x = torch.randn(128, 10)
        
        # Tạo 2 "view" (augmentation) từ cùng một dữ liệu gốc
        view_1 = x + 0.05 * torch.randn_like(x)
        view_2 = x + 0.05 * torch.randn_like(x)

        # Trích xuất biểu diễn (embeddings)
        q = encoder(view_1)
        k_pos = encoder(view_2)

        # Tính InfoNCE loss (dùng in-batch negatives)
        loss = info_nce(q, k_pos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            with torch.no_grad():
                q_norm = F.normalize(q, dim=-1)
                k_norm = F.normalize(k_pos, dim=-1)
                avg_pos_sim = (q_norm * k_norm).sum(dim=-1).mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Avg Positive Cosine Sim: {avg_pos_sim:.4f}")

    print("\n=> Huấn luyện thành công: Cosine similarity của các cặp cùng nguồn gốc tiệm cận 1.0!")
