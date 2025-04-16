# ComfyUI-latent-ops

ComfyUI で `LATENT` の操作を行うノード群です。

ノードはすべて `hnmr/latent_ops` 以下に追加されます。

# Nodes

| 名前 | 機能 |
| --- | --- |
| AssertDims | 入力された `LATENT` が指定された次元数であることを確認する |
| AssertShape | 入力された `LATENT` が指定された形状であることを確認する |
| LatentOperationNormalizeAlongAxis | 特定の軸に沿って正規化を行う |
| LatentOperationNormalize | 全体を正規化する |
| LatentOperationLayerNorm | レイヤー正規化を適用する |
| LatentOperationInstanceNorm | インスタンス正規化を適用する |
| LatentOperationNormalizeMinMax | 最小値と最大値の間で正規化する |
| LatentOperationNormalizePercentile | パーセンタイルに基づいて正規化する |
| LatentOperationSigmoid | シグモイド関数を適用する |
| LatentOperationHardSigmoid | ハードシグモイド関数を適用する |
| LatentOperationLogistic | ロジスティック関数を適用する |
| LatentOperationTanh | tanh関数を適用する |
| LatentOperationHardTanh | ハードtanh関数を適用する |
| LatentOperationSinh | sinh関数を適用する |
| LatentOperationCosh | cosh関数を適用する |
| LatentOperationReLU | ReLU関数を適用する |
| LatentOperationReLU6 | ReLU6関数を適用する |
| LatentOperationLeakyReLU | LeakyReLU関数を適用する |
| LatentOperationELU | ELU関数を適用する |
| LatentOperationSELU | SELU関数を適用する |
| LatentOperationCELU | CELU関数を適用する |
| LatentOperationGELU | GELU関数を適用する |
| LatentOperationSiLU | SiLU関数を適用する |
| LatentOperationHardSwish | HardSwish関数を適用する |
| LatentOperationMish | Mish関数を適用する |
| LatentOperationSoftplus | Softplus関数を適用する |
| LatentOperationSoftmax | Softmax関数を適用する |
| LatentOperationSoftmin | Softmin関数を適用する |
| LatentOperationSoftsign | Softsign関数を適用する |
| LatentOperationReshape | `LATENT`の形状を変更する |
| LatentOperationSlice | `LATENT`の一部を取り出す |
| LatentOperationRoll | `LATENT`をシフトさせる |
| LatentOperationAddBroadcast | 値をブロードキャストして加算する |
| LatentOperationMulBroadcast | 値をブロードキャストして乗算する |
| LatentOperationFill | 指定した値で埋める |
| LatentOperationAdd | 別の`LATENT`を加算する |
| LatentOperationMul | 別の`LATENT`を乗算する |
| LatentOperationClamp | 上限と下限で値を制限する |
| LatentOperationClampMin | 下限で値を制限する |
| LatentOperationClampMax | 上限で値を制限する |
| LatentOperationApplyCFG | CFG調整を適用する |
| LatentOperationSplitCFG | CFGを条件付き部分と無条件部分に分割する |
| LatentOperationInterpolate | 2つの`LATENT`間を補間する |
| Latent01ToImage | 0〜1範囲の`LATENT`を画像に変換する |
| Latent11ToImage | -1〜1範囲の`LATENT`を画像に変換する (`z = x * 0.5 + 0.5`) |
| GetSigma | SIGMASから特定のインデックスの σ を取得する |
