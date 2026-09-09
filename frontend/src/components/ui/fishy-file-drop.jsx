"use client";
import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion, useSpring } from "framer-motion";

const FishyFileDrop = ({
  id,
  width = "100%",
  height = "200px",
  padding = "16px",
  text = "Drop drone footage here",
  onFilesSelected,
}) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  // Realistic drone inertia physics (slow, smooth tracking like a heavy UAV)
  const droneX = useSpring(0, { stiffness: 25, damping: 22, mass: 1.5 });
  const droneY = useSpring(-20, { stiffness: 25, damping: 22, mass: 1.5 });
  const droneRotate = useSpring(0, { stiffness: 40, damping: 20 });

  // Store the anchor position where the drone was left by the cursor
  const lastPosRef = useRef({ x: 0, y: -20 });

  const handleMouseEnter = useCallback(() => setIsHovered(true), []);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
    // Lock current spring values as new anchor point so it doesn't snap or restart animation
    lastPosRef.current = {
      x: droneX.get(),
      y: droneY.get(),
    };
  }, [droneX, droneY]);

  const handleMouseMove = useCallback((e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;

    lastPosRef.current = { x, y: y - 25 };

    // Slowly glide towards cursor
    droneX.set(x);
    droneY.set(y - 25);
    droneRotate.set((x - droneX.get()) * 0.12); // Dynamic flight banking angle
  }, [droneX, droneY, droneRotate]);

  // Smooth continuous idle roaming starting from wherever the drone was left
  useEffect(() => {
    if (isHovered) return;

    let animFrame;
    let startTime = performance.now();
    const anchorX = lastPosRef.current.x;
    const anchorY = lastPosRef.current.y;

    const loop = (now) => {
      const elapsed = (now - startTime) / 1000;
      
      // Gentle sine/cosine floating around the last anchor point
      const offsetX = Math.sin(elapsed * 0.9) * 35;
      const offsetY = Math.cos(elapsed * 1.3) * 12;
      const rot = Math.sin(elapsed * 0.9) * 5;

      droneX.set(anchorX + offsetX);
      droneY.set(anchorY + offsetY);
      droneRotate.set(rot);

      animFrame = requestAnimationFrame(loop);
    };

    animFrame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animFrame);
  }, [isHovered, droneX, droneY, droneRotate]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFilesSelected(e.dataTransfer.files);
      }
    },
    [onFilesSelected]
  );

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFilesSelected(e.target.files);
    }
  };

  const onButtonClick = () => {
    inputRef.current?.click();
  };

  const isActive = isHovered || isDragActive;

  return (
    <div
      id={id}
      ref={containerRef}
      onClick={onButtonClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseMove={handleMouseMove}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      style={{
        width,
        height,
        padding,
        position: "relative",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: isActive ? "#0a0a0a" : "#000",
        border: isActive ? "1px solid #ededed" : "1px dashed #333",
        borderRadius: "8px",
        overflow: "hidden",
        cursor: "pointer",
        transition: "all 0.2s ease",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="video/mp4,video/x-m4v,video/quicktime,video/*"
        onChange={handleChange}
        style={{ display: "none" }}
      />

      {/* Heavy UAV Smooth Physics Drone Graphic */}
      <motion.div
        style={{
          x: droneX,
          y: droneY,
          rotate: droneRotate,
          position: "absolute",
          zIndex: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          pointerEvents: "none",
        }}
      >
        <svg
          width="48"
          height="48"
          viewBox="0 0 24 24"
          fill="none"
          stroke={isActive ? "#ededed" : "#666"}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transition: "stroke 0.2s ease" }}
        >
          {/* Drone Body & Center Sensor */}
          <path d="M12 10v4M10 12h4" />
          <circle cx="12" cy="12" r="2.5" />
          {/* Arms */}
          <line x1="10" y1="10" x2="6" y2="6" />
          <line x1="14" y1="10" x2="18" y2="6" />
          <line x1="10" y1="14" x2="6" y2="18" />
          <line x1="14" y1="14" x2="18" y2="18" />
          {/* Propeller Rings */}
          <motion.circle
            cx="5"
            cy="5"
            r="3"
            strokeDasharray="4 2"
            animate={{ rotate: 360 }}
            transition={{ duration: isHovered ? 0.3 : 0.8, repeat: Infinity, ease: "linear" }}
          />
          <motion.circle
            cx="19"
            cy="5"
            r="3"
            strokeDasharray="4 2"
            animate={{ rotate: -360 }}
            transition={{ duration: isHovered ? 0.3 : 0.8, repeat: Infinity, ease: "linear" }}
          />
          <motion.circle
            cx="5"
            cy="19"
            r="3"
            strokeDasharray="4 2"
            animate={{ rotate: -360 }}
            transition={{ duration: isHovered ? 0.3 : 0.8, repeat: Infinity, ease: "linear" }}
          />
          <motion.circle
            cx="19"
            cy="19"
            r="3"
            strokeDasharray="4 2"
            animate={{ rotate: 360 }}
            transition={{ duration: isHovered ? 0.3 : 0.8, repeat: Infinity, ease: "linear" }}
          />
        </svg>
      </motion.div>

      {/* Main Text Label */}
      <span
        style={{
          zIndex: 5,
          fontFamily: "var(--font-sans)",
          fontWeight: 500,
          fontSize: "13px",
          color: isActive ? "#ededed" : "#888",
          marginTop: "40px",
          transition: "color 0.2s ease",
        }}
      >
        {text}
      </span>

      {/* Subtext */}
      <span
        style={{
          zIndex: 5,
          marginTop: "4px",
          fontSize: "11px",
          fontFamily: "var(--font-sans)",
          color: "#666",
        }}
      >
        MP4, MOV up to 500MB
      </span>
    </div>
  );
};

export { FishyFileDrop, FishyFileDrop as DroneFileDrop };
