// Navigation Bar
// ------------
// Description: The navigation bar data for the website.
export interface Logo {
  src: string;
  alt: string;
  text: string;
}

export interface NavSubItem {
  name: string;
  link: string;
}

export interface NavItem {
  name: string;
  link: string;
  submenu?: NavSubItem[];
}

export interface NavAction {
  name: string;
  link: string;
  style: string;
  size: string;
}

export interface NavData {
  logo: Logo;
  navItems: NavItem[];
  navActions: NavAction[];
}

export const navigationBarData: NavData = {
  logo: {
    src: "/logo.svg",
    alt: "splinter logo",
    text: "",
  },
  navItems: [
    { name: "Home", link: "/" },
    { name: "Features", link: "#highlight-0" },
    { name: "Team", link: "/teamPage" },
  ],
  navActions: [
    { name: "Case Study", link: "/case-study", style: "primary", size: "lg" },
  ],
};
