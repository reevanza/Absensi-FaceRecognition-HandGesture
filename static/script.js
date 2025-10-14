const video = document.getElementById("video");
const statusText = document.getElementById("status");

const instruksiBox = document.createElement("div");
instruksiBox.id = "instruksi";
instruksiBox.style.marginTop = "15px";
instruksiBox.style.fontSize = "18px";
instruksiBox.style.fontWeight = "600";
document.body.insertBefore(instruksiBox, statusText);

// Ambil instruksi dari server
async function getInstruction() {
  const res = await fetch("/instruction");
  const data = await res.json();
  instruksiBox.innerHTML = `
    <span>🧠 Instruksi:</span><br>
    <b>Mulut:</b> ${data.mouth}<br>
    <b>Gesture:</b> ${data.gesture}
  `;
}
getInstruction(); // panggil sekali saat halaman dimuat

// ==== KODE LAMA (kirim frame ke server) ====
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => console.error("Tidak bisa akses kamera:", err));

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

async function sendFrame() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg");

  const res = await fetch("/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl })
  });

  const data = await res.json();
  statusText.textContent = `Status: ${data.status} (${data.name})`;
}

setInterval(sendFrame, 2000);
