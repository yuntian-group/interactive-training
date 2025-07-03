import favicon from '../../assets/favicon.ico';
import clsx from "clsx";

type Props = React.HTMLAttributes<HTMLDivElement>;

const NavigationBar: React.FC<Props> = ({ className }: Props) => {
    return (
        <div className={clsx("bg-black text-white w-full flex items-center p-2 sticky top-0", className)}>
            <img src={favicon} alt="Favicon" className="inline-block h-8 w-8 ml-2" />
            <span className="ml-4 text-2xl">Interactive Training</span>
        </div>
    );
};

export default NavigationBar;