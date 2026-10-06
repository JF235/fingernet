// O template mufis v1 escrito DE DENTRO DO GRAFO, do batch que está em memória.
//
// Antes, um template só saía lendo de volta os PNGs que o sink tinha acabado de escrever --
// mufv1 só existia em Python. Duas consequências, e a segunda é a que importa:
//
//   * gravação DUPLA: 883 MB de produtos para produzir 28 MB de template;
//   * e o bloco de qualidade era uma reamostragem de um upsample. A rede calcula numa
//     grade stride-8; `quality_u8` faz o bilinear dela para 512x512; o conversor Python
//     voltava a 64x64 estimando a grade a partir daquele bilinear (a média dos pixels 3 e
//     4 de cada célula, que é o melhor palpite possível ali). Medido: 1,13/255 de erro
//     médio por pixel dentro da máscara.
//
// Aqui a grade É o que a rede produziu. `FnetRaw` carrega `segmentation` e
// `orientation_index` em h x w, então os dois rasters saem da fonte:
//
//   mask         `cleaned` do MaskProduct, que já é a grade binária
//   quality      trunc(segmentation * 255), o MESMO mapeamento que quality_u8 aplica --
//                aplicado no ponto da grade em vez de depois do bilinear
//   orientation  lround(bin_to_angle(idx) * 180/PI + 90), que é exatamente o byte por
//                célula que orientation_png calcula antes de expandir
//
// A IDENTIDADE vem do nome do arquivo (`{dataset}_{IID}_{FID}-{SID}`), a mesma gramática
// que o `FID_RE` do lado Python lê. Um nome fora dela vira uma identidade de um dedo, em
// vez de ser descartado: melhor um template por imagem que nenhum.
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "arandu_nodes.hpp"   // RawImage: a grade da rede vem dele
#include "minfmt.hpp"         // to_min: as minúcias no domínio do .min
#include "mufv1/bundle.hpp"
#include "mufv1/fingernet.hpp"
#include "mufv1/mufv1.hpp"
#include "postproc.hpp"

namespace fnaru::mufv1out {

namespace fs = std::filesystem;

/// Os modos de saída. Um modo, uma pasta, nenhum byte escrito duas vezes.
///
///   products      <out>/<produto>/<id>.png|.min      os cinco produtos, e nenhum template
///   bundle        <out>/templates.mufi               tudo num arquivo, + o manifest
///   per_image     <out>/<id>.mufi                    um por imagem
///   per_identity  <out>/<iid>/<sid>.mufi             um por registro (iid, sid)
enum class Mode { Products, Bundle, PerImage, PerIdentity };

inline Mode mode_of(const std::string& s) {
    if (s == "products") return Mode::Products;
    if (s == "bundle") return Mode::Bundle;
    if (s == "per-image") return Mode::PerImage;
    if (s == "per-identity") return Mode::PerIdentity;
    throw std::runtime_error("modo de saída desconhecido: '" + s +
                             "' (products | bundle | per-image | per-identity)");
}

/// (dataset, iid, sid, fid) lidos do nome, ou o nome inteiro como identidade quando ele não
/// segue a gramática. Espelha `bases.FID_RE`: `{dataset}_{IID}_{FID}-{SID}`, com o FID de
/// um ou dois dígitos e um sufixo livre depois do SID.
struct Ident {
    std::string dataset, iid, sid;
    int fid = 11;                      // 11 = desconhecido, a convenção do projeto
};

inline Ident ident_of(const std::string& id) {
    // `id` pode ter pastas dentro (`0001/01/sd4_0001_01-01_whirl`); a gramática está no
    // ÚLTIMO componente.
    const std::size_t slash = id.find_last_of('/');
    const std::string stem = slash == std::string::npos ? id : id.substr(slash + 1);

    Ident out;
    out.dataset = "graph";
    out.iid = stem;
    out.sid = "00";

    // dataset_IID_FID-SID[_extra]
    const std::size_t u1 = stem.find('_');
    if (u1 == std::string::npos) return out;
    const std::size_t u2 = stem.find('_', u1 + 1);
    if (u2 == std::string::npos) return out;
    const std::size_t dash = stem.find('-', u2 + 1);
    if (dash == std::string::npos) return out;
    const std::string ds = stem.substr(0, u1);
    const std::string iid = stem.substr(u1 + 1, u2 - u1 - 1);
    const std::string fid = stem.substr(u2 + 1, dash - u2 - 1);
    std::size_t end = dash + 1;
    while (end < stem.size() && std::isdigit(static_cast<unsigned char>(stem[end]))) ++end;
    const std::string sid = stem.substr(dash + 1, end - dash - 1);

    auto digits = [](const std::string& s) {
        return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        });
    };
    if (!digits(iid) || !digits(fid) || !digits(sid)) return out;

    out.dataset = ds;
    for (char& c : out.dataset) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    out.iid = iid;
    out.sid = sid;
    out.fid = std::stoi(fid);
    return out;
}

/// Um dedo pronto para virar bloco: a grade da rede, já em domínio armazenado.
struct FingerData {
    mufv1::Finger meta;
    std::vector<std::uint8_t> mask, quality, orientation;   // rows x cols, linha por linha
    std::vector<std::vector<double>> mnt;                   // 5 colunas
    int rows = 0, cols = 0;
    std::string id;                                         // o id do runner, para o nome
};

/// A grade que a rede produziu, sem passar por 512x512.
inline FingerData coarse_of(const RawImage& r, const std::vector<float>& cleaned,
                            const std::vector<fnpost::Minutia>& mnt, const std::string& id) {
    FingerData fd;
    fd.rows = r.h;
    fd.cols = r.w;
    fd.id = id;
    const std::size_t n = static_cast<std::size_t>(r.h) * r.w;

    fd.mask.resize(n);
    for (std::size_t i = 0; i < n; ++i) fd.mask[i] = cleaned[i] > 0.5f ? 1 : 0;

    // O MESMO mapeamento de quality_u8 (truncamento para u8 depois de x255), aplicado no
    // ponto da grade em vez de depois do bilinear.
    fd.quality.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        fd.quality[i] = fnpost::to_u8_trunc(r.segmentation[i] * 255.0f);

    // E o byte por célula que orientation_png calcula antes de expandir.
    fd.orientation.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float a = fnpost::bin_to_angle(r.orientation_index[i]);
        const long v = std::lround(a * 180.0 / fnpost::PI + 90.0);
        fd.orientation[i] = static_cast<std::uint8_t>(std::clamp<long>(v, 0, 255));
    }

    fd.mnt.assign(5, {});
    for (const auto& m : mnt) {
        const fnmin::MinRow row = fnmin::to_min(m);
        fd.mnt[0].push_back(static_cast<double>(row.x));
        fd.mnt[1].push_back(static_cast<double>(row.y));
        fd.mnt[2].push_back(static_cast<double>(row.angle));
        fd.mnt[3].push_back(static_cast<double>(row.quality));
        fd.mnt[4].push_back(0.0);      // type: a fingernet não classifica
    }
    return fd;
}

/// Os quatro blocos, na ordem em que fingernet.py os adiciona -- que é a ordem no schema, e
/// o `support` de um bloco aponta para um id que já tem de existir. O QUE cada bloco é vive
/// em mufv1/fingernet.hpp, declarado uma vez; aqui só se diz com que dados preenchê-los.
inline mufv1::Record record_of(const Ident& idt, const std::vector<FingerData>& fingers,
                               const std::string& theta_dtype) {
    mufv1::Record rec(idt.dataset, idt.iid, idt.sid);
    for (const auto& f : fingers) rec.add_finger(f.meta);

    std::vector<std::vector<std::vector<double>>> mnt;
    for (const auto& f : fingers) mnt.push_back(f.mnt);
    rec.add_points(mufv1::fnet::minutiae_block(theta_dtype), mnt);

    const mufv1::Grid g = mufv1::fnet::grid_of(fingers.empty() ? 0 : fingers[0].rows,
                                               fingers.empty() ? 0 : fingers[0].cols);
    std::vector<std::vector<std::uint8_t>> mask, quality, ori;
    for (const auto& f : fingers) {
        mask.push_back(f.mask);
        quality.push_back(f.quality);
        ori.push_back(f.orientation);
    }
    rec.add_raster(mufv1::fnet::mask_block(g), mask);
    rec.add_raster(mufv1::fnet::quality_block(g), quality, mask);
    rec.add_raster(mufv1::fnet::orientation_block(g), ori, mask);
    return rec;
}

/// O que uma corrida escreveu, e o que ela não precisou escrever.
struct Tally {
    std::size_t written = 0;
    std::size_t skipped = 0;      // já estavam no bundle (a regra 3 do formato)
};

/// Onde os templates vão parar, nos três modos que os escrevem.
///
/// O SINK É COMPARTILHADO ENTRE OS ATENDENTES do JOIN (o `run` é const e vários o chamam),
/// então tudo aqui é guardado por mutex. No bundle isso serializa a escrita, que é o que se
/// quer: um arquivo, um fsync por registro, e um manifest que não intercala linhas.
class Writer {
public:
    Writer(Mode mode, fs::path out, std::string theta_dtype)
        : mode_(mode), out_(std::move(out)), theta_(std::move(theta_dtype)) {}

    /// PER_IDENTITY guarda os dedos até o FIM DA CORRIDA (ver `finish`): agrupar por
    /// (iid, sid) exige saber que todos chegaram, e um sink não tem como saber. O custo é
    /// memória proporcional às identidades DA CORRIDA -- ~9 kB de grade por dedo, 36 MB
    /// para as 4000 da SD4. Os outros dois modos escrevem na chegada e não guardam nada.
    ~Writer() {
        try {
            finish();
        } catch (...) {                 // um destrutor não propaga; a corrida já terminou
        }
    }

    void put(const Ident& idt, FingerData fd) {
        std::lock_guard<std::mutex> lk(mx_);
        switch (mode_) {
            case Mode::Bundle: {
                if (!bundle_) {
                    fs::create_directories(out_);
                    bundle_ = std::make_unique<mufv1::Writer>(out_ / "templates.mufi");
                }
                // A REGRA 3 EM USO: o manifest diz o que não refazer. Sem esta consulta,
                // rodar duas vezes sobre o mesmo bundle escrevia os mesmos registros de
                // novo -- 12 linhas para 6 identidades, e um índice com a mesma identidade
                // duas vezes. Quem quer começar de zero apaga o destino.
                //
                // A CHAVE INCLUI O DEDO, porque neste modo o registro é UMA IMAGEM. Sem o
                // fid, o segundo dedo da mesma amostra parecia já escrito: 100 imagens da
                // TS1k (dez dedos por (iid, sid)) viravam 10 registros, e qual dos dez
                // sobrava era quem chegasse primeiro no JOIN. Os outros 90 eram descartados
                // como "já estavam lá".
                if (bundle_->has(idt.iid, idt.sid, fd.meta.fid)) {
                    ++skipped_;
                    break;
                }
                std::vector<FingerData> one{std::move(fd)};
                bundle_->append(record_of(idt, one, theta_), idt.dataset, idt.iid, idt.sid);
                ++written_;
                break;
            }
            case Mode::PerImage: {
                std::vector<FingerData> one{std::move(fd)};
                const fs::path dest = out_ / (one[0].id + ".mufi");
                mufv1::write_record(dest, record_of(idt, one, theta_).to_bytes());
                ++written_;
                break;
            }
            case Mode::PerIdentity:
                pending_[{idt.iid, idt.sid}].first = idt;
                pending_[{idt.iid, idt.sid}].second.push_back(std::move(fd));
                break;
            case Mode::Products:
                break;                  // não é deste writer
        }
    }

    /// O FIM DE UMA CORRIDA, e não o fim do objeto.
    ///
    /// Um destrutor não serve aqui: com `graph_run --serve` o grafo fica QUENTE entre Runs,
    /// então o nó (e este writer) sobrevive à corrida — o per-identity nunca esvaziaria os
    /// pendentes, e o bundle ficaria com um descritor aberto num arquivo que o Run seguinte
    /// apaga (a pasta de rascunho é limpa antes de cada um). Chamado pelo `graph_run`
    /// depois de o executor voltar; o bundle reabre na próxima escrita.
    /// O que ESTA corrida fez, e zera a conta. Com o grafo quente o mesmo writer serve
    /// várias corridas, e um contador que acumula reportaria 12 na segunda passada de 6
    /// imagens. `skipped` é o que o manifest do bundle já tinha: é a diferença entre "nada
    /// aconteceu" e "nada precisava acontecer".
    Tally finish() {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto& [key, val] : pending_) {
            const fs::path dest = out_ / key.first / (key.second + ".mufi");
            mufv1::write_record(dest, record_of(val.first, val.second, theta_).to_bytes());
            ++written_;
        }
        pending_.clear();
        bundle_.reset();                // fecha; a próxima escrita reabre no destino atual
        const Tally t{written_, skipped_};
        written_ = 0;
        skipped_ = 0;
        return t;
    }

private:
    Mode mode_;
    fs::path out_;
    std::string theta_;
    mutable std::mutex mx_;
    std::unique_ptr<mufv1::Writer> bundle_;
    std::map<std::pair<std::string, std::string>,
             std::pair<Ident, std::vector<FingerData>>> pending_;
    std::size_t written_ = 0;
    std::size_t skipped_ = 0;
};

}  // namespace fnaru::mufv1out
