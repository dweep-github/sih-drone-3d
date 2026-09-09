import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, PerformanceMonitor } from '@react-three/drei';
import { Upload, Play, Terminal, Circle, FileVideo, HardDrive, Cpu, Loader2 } from 'lucide-react';
import * as THREE from 'three';
import { FishyFileDrop } from './components/ui/fishy-file-drop';

// R3F Performance Best Practices: Pre-allocate geometries and materials outside or via useMemo
function OptimizedDroneMesh() {
  const meshRef = useRef();
  
  const { bodyGeo, armGeo, mainMat, darkMat } = useMemo(() => {
    return {
      bodyGeo: new THREE.BoxGeometry(2, 0.5, 1),
      armGeo: new THREE.CylinderGeometry(0.1, 0.1, 1),
      mainMat: new THREE.MeshStandardMaterial({ color: '#ededed', roughness: 0.6 }),
      darkMat: new THREE.MeshStandardMaterial({ color: '#333333', roughness: 0.8 })
    };
  }, []);

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <group ref={meshRef}>
      <mesh geometry={bodyGeo} material={mainMat} position={[0, 0, 0]} castShadow receiveShadow />
      <mesh geometry={armGeo} material={darkMat} position={[0.8, 0, 0.8]} castShadow />
      <mesh geometry={armGeo} material={darkMat} position={[-0.8, 0, -0.8]} castShadow />
      <mesh geometry={armGeo} material={darkMat} position={[0.8, 0, -0.8]} castShadow />
      <mesh geometry={armGeo} material={darkMat} position={[-0.8, 0, 0.8]} castShadow />
    </group>
  );
}

// Shadcn UI minimal components
const Card = ({ children, className = "" }) => (
  <div className={`bg-[#000] geist-border rounded-lg overflow-hidden flex flex-col ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children, className = "" }) => (
  <div className={`px-4 py-3 border-b border-[#333] ${className}`}>
    {children}
  </div>
);

const CardTitle = ({ children, className = "" }) => (
  <h3 className={`text-sm font-semibold tracking-tight ${className}`}>
    {children}
  </h3>
);

const CardContent = ({ children, className = "" }) => (
  <div className={`p-4 ${className}`}>
    {children}
  </div>
);

const Button = ({ children, variant = "default", className = "", ...props }) => {
  const baseStyle = "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-ring disabled:pointer-events-none disabled:opacity-50 h-9 px-4 py-2";
  const variants = {
    default: "bg-[#ededed] text-black hover:bg-[#ededed]/90",
    outline: "border border-[#333] bg-transparent hover:bg-[#111] text-[#ededed]",
    ghost: "hover:bg-[#111] text-[#ededed]",
  };
  return (
    <button className={`${baseStyle} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
};

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [logs, setLogs] = useState([{ msg: "System initialized.", type: "INFO", time: Date.now() }]);
  const [isUploading, setIsUploading] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const logsEndRef = useRef(null);
  const [dpr, setDpr] = useState(1);

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  const addLog = (msg, type = "INFO") => {
    setLogs((prev) => [...prev, { msg, type, time: Date.now() }]);
  };

  const handleFilesSelected = (files) => {
    if (files && files[0]) {
      const file = files[0];
      setVideoFile(file);
      addLog(`Drone video attached: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`, "INFO");
    }
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    setIsUploading(true);
    addLog(`Uploading drone video (${videoFile.name})...`, "PROCESS");
    
    try {
      const formData = new FormData();
      formData.append('file', videoFile);
      // const response = await axios.post('http://127.0.0.1:8000/api/upload-video', formData);
      await new Promise(r => setTimeout(r, 1000));
      addLog("Upload successful.", "SUCCESS");
    } catch (error) {
      addLog("Upload failed. Mock fallback used.", "WARNING");
    } finally {
      setIsUploading(false);
    }
  };

  const runPipeline = async () => {
    setIsRunning(true);
    addLog("Starting reconstruction pipeline...", "PROCESS");
    addLog("Extracting frames...", "PROCESS");
    
    try {
      // const response = await axios.post('http://127.0.0.1:8000/api/run-pipeline');
      await new Promise(r => setTimeout(r, 1500));
      addLog("Extraction: 356 clean frames isolated.", "SUCCESS");
      addLog("Initiating YOLOv11 masking...", "PROCESS");
      await new Promise(r => setTimeout(r, 1500));
      addLog("Masking completed. 98% confidence.", "SUCCESS");
      addLog("Pipeline complete.", "SUCCESS");
    } catch (error) {
      addLog("Pipeline failed.", "ERROR");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-[#ededed] flex flex-col font-sans selection:bg-[#333] selection:text-[#fff]">
      {/* Header */}
      <header className="h-14 border-b border-[#333] px-6 flex items-center justify-between shrink-0 bg-black z-10">
        <div className="flex items-center gap-3">
          <HardDrive className="w-5 h-5 text-[#ededed]" />
          <h1 className="text-sm font-semibold tracking-tight">
            UAV Reconstruction
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <Circle className="w-2 h-2 fill-green-500 text-green-500" />
          <span className="text-xs font-medium text-[#888]">System Online</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col lg:flex-row p-6 gap-6 overflow-hidden h-[calc(100vh-56px)]">
        
        {/* Left Panel */}
        <div className="w-full lg:w-[400px] flex flex-col gap-6 shrink-0 h-full">
          
          <Card className="shrink-0">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="w-4 h-4" /> Control Panel
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-5">
              
              <div className="space-y-3">
                <label className="text-xs font-medium text-[#888] uppercase tracking-wider">Drone Video Dropzone</label>
                <div className="flex flex-col items-center justify-center">
                  <FishyFileDrop 
                    width="100%" 
                    height="180px" 
                    padding="15px"
                    text={videoFile ? videoFile.name : "Drop Video"}
                    textSize="14px"
                    textHoverSize="16px"
                    letterSpacing="2px"
                    letterSpacingHover="4px"
                    borderColor="#333"
                    borderRadius="12px"
                    innerBorderRadius="8px"
                    waveColors={["#1b70a1", "#368cc1", "#50a8e0", "#6bc4ff"]}
                    onFilesSelected={handleFilesSelected}
                  />
                  {videoFile && (
                    <div className="w-full flex items-center justify-between mt-2 px-1">
                      <span className="text-xs text-[#888] truncate max-w-[220px]">
                        {videoFile.name}
                      </span>
                      <Button 
                        variant="outline" 
                        onClick={handleUpload}
                        disabled={isUploading}
                        className="h-7 px-3 text-xs"
                      >
                        {isUploading ? <Loader2 className="w-3 h-3 animate-spin mr-1" /> : <Upload className="w-3 h-3 mr-1" />}
                        Upload
                      </Button>
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-3">
                <label className="text-xs font-medium text-[#888] uppercase tracking-wider">Execution</label>
                <Button 
                  onClick={runPipeline}
                  disabled={isRunning}
                  className="w-full justify-start pl-3"
                >
                  {isRunning ? (
                    <><Loader2 className="w-4 h-4 animate-spin mr-2" /> Processing Pipeline...</>
                  ) : (
                    <><Play className="w-4 h-4 mr-2" /> Run Reconstruction</>
                  )}
                </Button>
              </div>
              
              <div className="grid grid-cols-2 gap-px bg-[#333] rounded-md overflow-hidden border border-[#333]">
                <div className="bg-[#000] p-3 flex flex-col">
                  <span className="text-[#888] text-xs font-medium">Frames</span>
                  <span className="text-xl font-semibold tracking-tight mt-1">356</span>
                </div>
                <div className="bg-[#000] p-3 flex flex-col">
                  <span className="text-[#888] text-xs font-medium">Masking</span>
                  <span className="text-xl font-semibold tracking-tight mt-1">98%</span>
                </div>
              </div>

            </CardContent>
          </Card>

          <Card className="flex-1 min-h-[200px]">
            <CardHeader className="py-2.5 bg-[#050505]">
              <CardTitle className="text-xs flex items-center gap-2 text-[#888] font-mono">
                <Terminal className="w-3.5 h-3.5" /> Output
              </CardTitle>
            </CardHeader>
            <div className="flex-1 p-3 overflow-y-auto font-mono text-[11px] leading-relaxed space-y-1 bg-[#000] h-[calc(100%-41px)]">
              {logs.map((log, i) => {
                const timeStr = new Date(log.time).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' });
                return (
                  <div key={i} className="flex gap-3 hover:bg-[#111] px-1 rounded -mx-1">
                    <span className="text-[#666] shrink-0">{timeStr}</span>
                    <span className={`
                      ${log.type === 'ERROR' ? 'text-red-400' : ''}
                      ${log.type === 'SUCCESS' ? 'text-[#ededed]' : ''}
                      ${log.type === 'WARNING' ? 'text-yellow-400' : ''}
                      ${log.type === 'INFO' ? 'text-[#888]' : ''}
                      ${log.type === 'PROCESS' ? 'text-[#888]' : ''}
                    `}>
                      {log.msg}
                    </span>
                  </div>
                );
              })}
              <div ref={logsEndRef} />
            </div>
          </Card>
        </div>

        {/* Right Panel: Viewport */}
        <Card className="flex-1 relative overflow-hidden">
          <div className="absolute top-4 left-4 z-10 bg-[#000]/80 backdrop-blur border border-[#333] px-3 py-1.5 rounded-md flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#ededed] animate-pulse" />
            <span className="text-[11px] font-medium tracking-wider uppercase text-[#888]">Viewport 3D</span>
          </div>
          
          <div className="flex-1 w-full h-full bg-[#111] cursor-grab active:cursor-grabbing">
            <Canvas 
              shadows 
              dpr={dpr} 
              gl={{ antialias: false, powerPreference: "high-performance" }}
              camera={{ position: [5, 5, 5], fov: 45, near: 0.1, far: 1000 }}
            >
              <PerformanceMonitor onDecline={() => setDpr(1)} onIncline={() => setDpr(2)} />
              <color attach="background" args={["#111111"]} />
              
              <ambientLight intensity={0.4} />
              <directionalLight position={[10, 10, 5]} intensity={1} castShadow shadow-mapSize={[1024, 1024]} />
              
              <OptimizedDroneMesh />
              
              <Grid infiniteGrid fadeDistance={20} sectionColor="#444" cellColor="#222" sectionSize={1} cellSize={0.2} />
              <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
            </Canvas>
          </div>
        </Card>
      </main>
    </div>
  );
}

export default App;
